############################################################################
# Copyright (C) 2025, Advanced Micro Devices, Inc.
# All rights reserved.
#
# SPDX-License-Identifier: MIT 
#
# @author       Shane T. Fleming <shane.fleming@amd.com>
############################################################################

import pytest
import torch
import torch.onnx
from torch import nn
import onnx
import tempfile
import numpy as np
import os


from qonnx.core.datatype import DataType
from qonnx.util.basic import gen_finn_dt_tensor, qonnx_make_model
from onnx import helper, TensorProto
from qonnx.core.modelwrapper import ModelWrapper
from qonnx.transformation.infer_shapes import InferShapes
from qonnx.transformation.infer_datatypes import InferDataTypes
from qonnx.transformation.general import GiveReadableTensorNames, GiveUniqueNodeNames, ApplyConfig
from qonnx.core.modelwrapper import ModelWrapper
from brevitas.export import export_qonnx
from qonnx.util.cleanup import cleanup as qonnx_cleanup

import finn.core.onnx_exec as oxe
from finn.transformation.fpgadataflow.set_exec_mode import SetExecMode
from finn.transformation.fpgadataflow.specialize_layers import SpecializeLayers
from finn.transformation.fpgadataflow.prepare_cppsim import PrepareCppSim
from finn.transformation.fpgadataflow.compile_cppsim import CompileCppSim
from finn.transformation.fpgadataflow.prepare_ip import PrepareIP
from finn.transformation.fpgadataflow.hlssynth_ip import HLSSynthIP
from finn.transformation.fpgadataflow.prepare_rtlsim import PrepareRTLSim
from finn.transformation.fpgadataflow.create_stitched_ip import CreateStitchedIP
from finn.transformation.fpgadataflow.synth_ooc import SynthOutOfContext

from finn.transformation.fpgadataflow.shuffle_helpers import shuffle_perfect_loopnest_coeffs
from finn.transformation.fpgadataflow.convert_to_hw_layers import InferShuffle
from finn.transformation.fpgadataflow.transpose_decomposition import TransposeDecomposition
from qonnx.transformation.base import Transformation

test_fpga_part:str = "xcv80-lsva4737-2MHP-e-S"
test_synth_clk_period_ns:int = 5

class PytorchShuffle(nn.Module):
    """ From pytorch create a reshape and transpose combination
    that can be used for testing """

    def __init__(self, transpose_perm:tuple[int], 
            reshape1_shape:tuple[int]=None, 
            reshape2_shape:tuple[int]=None
        )->None:
        super(PytorchShuffle, self).__init__()
        self.transpose_perm = transpose_perm
        self.reshape1_shape = reshape1_shape
        self.reshape2_shape = reshape2_shape

    def forward(self, x):
        if self.reshape1_shape is not None:
            x = x.reshape(*self.reshape1_shape)
        x = x.permute(*self.transpose_perm)
        if self.reshape2_shape is not None:
            x = x.reshape(*self.reshape2_shape)
        return x

def construct_onnx_model(
        input_shape:tuple[int],
        transpose_perm:tuple[int],
        reshape1_shape:tuple[int],
        reshape2_shape:tuple[int],
        dt:DataType,
    )->ModelWrapper:

    """ Creates an ONNX model that can be used for testing
    the shuffle operation compiler integration. Uses the 
    pytorch methods in PytorchShuffle to generate the model. """

    dummy_input = torch.randn(*input_shape)
    model = PytorchShuffle(
            transpose_perm=transpose_perm,
            reshape1_shape=reshape1_shape,
            reshape2_shape=reshape2_shape
    )

    with tempfile.NamedTemporaryFile(delete=False, suffix=".onnx") as temp_file:
        model_input = torch.rand(input_shape)
        export_qonnx(model, model_input, temp_file.name, opset_version=17) 
        qonnx_cleanup(temp_file.name, out_file=temp_file.name)

        new_model = ModelWrapper(temp_file.name)
        new_model.set_tensor_datatype(new_model.graph.input[0].name, dt)
        new_model.set_tensor_datatype(new_model.graph.output[0].name, dt)
        new_model.transform(InferShapes())
        new_model.transform(InferDataTypes())
        return new_model
    raise RuntimeError(f"Error unable to export the ONNX file to the temporary location")


class SetShuffleSIMD(Transformation):
    """Set SIMD parameter and enable waveform generation for all Shuffle and PTranspose nodes."""
    
    def __init__(self, simd_value, enable_waveforms=False):
        super().__init__()
        self.simd_value = simd_value
        self.enable_waveforms = enable_waveforms
    
    def apply(self, model):
        graph_modified = False
        for node in model.graph.node:
            if node.op_type in ["Shuffle_hls", "PTranspose_rtl"] and "finn.custom_op.fpgadataflow" in node.domain:
                simd_found = False
                for attr in node.attribute:
                    if attr.name == "SIMD":
                        attr.i = self.simd_value
                        simd_found = True
                        break
                if not simd_found:
                    simd_attr = helper.make_attribute("SIMD", self.simd_value)
                    node.attribute.append(simd_attr)
                
                # Enable waveform generation for debugging
                if self.enable_waveforms:
                    trace_found = False
                    for attr in node.attribute:
                        if attr.name == "rtlsim_trace":
                            attr.i = "debug.wdb"
                            trace_found = True
                            break
                    if not trace_found:
                        trace_attr = helper.make_attribute("rtlsim_trace", "debug.wdb")
                        node.attribute.append(trace_attr)
        return model, False 

class SetCppSimExec(Transformation):
    """Set Exec mode for only HLS nodes"""
    
    def __init__(self):
        super().__init__()
    
    def apply(self, model):
        graph_modified = False
        for node in model.graph.node:
            if node.op_type in ["Shuffle_hls"] and "finn.custom_op.fpgadataflow" in node.domain:
                exec_mode_found = False
                for attr in node.attribute:
                    if attr.name == "exec_mode":
                        attr.i = "cppsim"
                        exec_mode_found = True
                        break
                if not exec_mode_found:
                    exec_mode_attr = helper.make_attribute("exec_mode", "cppsim")
                    node.attribute.append(exec_mode_attr)
                
        return model, False 


@pytest.mark.parametrize("cpp_shuffle_param", [ 
    {
            "in_shape" : (1,128,384), # Shuffle A
            "in_reshaped" : (1,128,12,32),
            "out_shape" : (1,12,128,32),
            "out_reshaped" : None,
            "perm" : (0,2,1,3)
    }, 
    {
            "in_shape" : (1,128,384), # Shuffle B 
            "in_reshaped" : (1,128,12,32),
            "out_shape" : (1,12,32,128),
            "out_reshaped" : None,
            "perm" : (0,2,3,1)
    }, 
    {
            "in_shape" : (4,8,4), # Brute Force cannot be simplified into 2D case 
            "in_reshaped" : None,
            "out_shape" : (4,8,4),
            "out_reshaped" : None,
            "perm" : (2,1,0)
    }, 
    {
            "in_shape" : (2,4,3), # Brute Force cannot be simplified into 2D case 
            "in_reshaped" : None,
            "out_shape" : (2,3,4),
            "out_reshaped" : None,
            "perm" : (0,2,1)
    }, 
    {
            "in_shape" : (1,12,128,32), # Shuffle C 
            "in_reshaped" : None,
            "out_shape" : (1,128,12,32),
            "out_reshaped" : (1,128,384),
            "perm" : (0,2,1,3)
    }, 
])
@pytest.mark.parametrize("datatype", ["INT8", "INT4"])
@pytest.mark.parametrize("simd", ["simd1", "simd2", "simd4"])
@pytest.mark.fpgadataflow
def test_cppsim_shuffle_layer(cpp_shuffle_param, datatype, simd):
    ''' Checks cppsim of the shuffle_hls layer '''
    dt = DataType[datatype]
    simd = int(simd[-1])
    in_shape = cpp_shuffle_param["in_shape"]

    model = construct_onnx_model(
            input_shape=in_shape,
            transpose_perm=cpp_shuffle_param["perm"],
            reshape1_shape=cpp_shuffle_param["in_reshaped"],
            reshape2_shape=cpp_shuffle_param["out_reshaped"],
            dt=dt
    )

    input = gen_finn_dt_tensor(dt, in_shape)
    in_name = model.graph.input[0].name
    out_name = model.graph.output[0].name
    input_t = {in_name : input}

    # Get a reference for the shuffle 
    y_ref = oxe.execute_onnx(model, input_t)[out_name]

    # Attempt to build the HLS for this
    model = model.transform(TransposeDecomposition())
    model = model.transform(InferShuffle())
    model = model.transform(SpecializeLayers(test_fpga_part))
    model = model.transform(GiveUniqueNodeNames())
    model = model.transform(GiveReadableTensorNames())

    model = model.transform(SetShuffleSIMD(simd))
    model = model.transform(SetCppSimExec())
    model = model.transform(PrepareCppSim())
    model = model.transform(CompileCppSim())

    y_hw = oxe.execute_onnx(model, input_t)[out_name]

    y_hw_flat = y_hw.flatten()
    y_ref_flat = y_ref.flatten()
    for i in range(len(y_hw_flat)):
        if not np.allclose(y_hw_flat[i], y_ref_flat[i]):
            print(f"Index {i}, Expected {y_ref_flat[i]} -- Got {y_hw_flat[i]}")

    assert np.allclose(y_ref, y_hw), "Model output does not match expected output"


@pytest.mark.parametrize("shuffle_param", [ 
    {
            "in_shape" : (1,128,384), # Shuffle A
            "in_reshaped" : (1,128,12,32),
            "out_shape" : (1,12,128,32),
            "out_reshaped" : None,
            "perm" : (0,2,1,3)
    }, 
    {
            "in_shape" : (1,12,128,32), # Shuffle C 
            "in_reshaped" : None,
            "out_shape" : (1,128,12,32),
            "out_reshaped" : (1,128,384),
            "perm" : (0,2,1,3)
    }, 
    {
            "in_shape" : (128,384), # pTranspose Test 
            "in_reshaped" : None,
            "out_shape" : (384,128),
            "out_reshaped" : None,
            "perm" : (1,0)
    }, 
    {
            "in_shape" : (32,16,8,12), # Mixed Transpose test 
            "in_reshaped" : None,
            "out_shape" : (8,12,32,16),
            "out_reshaped" : None,
            "perm" : (2,3,0,1)
    }, 
    {
            "in_shape" : (2,2,12,8),  
            "in_reshaped" : None,
            "out_shape" : (2,2,8,12),
            "out_reshaped" : None,
            "perm" : (0,1,3,2)
    }, 
    {
            "in_shape" : (32,16,12,8), # Mixed Transpose test 
            "in_reshaped" : None,
            "out_shape" : (8,12,16,32),
            "out_reshaped" : None,
            "perm" : (3,2,1,0)
    },
    {
            "in_shape" : (64,256), 
            "in_reshaped" : None,
            "out_shape" : (256,64),
            "out_reshaped" : None,
            "perm" : (1,0)
    },
    {
            "in_shape" : (512,128), 
            "in_reshaped" : None,
            "out_shape" : (128,512),
            "out_reshaped" : None,
            "perm" : (1,0)
    },
    {
            "in_shape" : (256,512), 
            "in_reshaped" : None,
            "out_shape" : (512,256),
            "out_reshaped" : None,
            "perm" : (1,0)
    },
    {
            "in_shape" : (8,16,32), 
            "in_reshaped" : None,
            "out_shape" : (32,16,8),
            "out_reshaped" : None,
            "perm" : (2,1,0)
    },
    {
            "in_shape" : (4,64,128), 
            "in_reshaped" : None,
            "out_shape" : (64,4,128),
            "out_reshaped" : None,
            "perm" : (1,0,2)
    },
    {
            "in_shape" : (16,8,64), 
            "in_reshaped" : None,
            "out_shape" : (64,16,8),
            "out_reshaped" : None,
            "perm" : (2,0,1)
    },
    {
            "in_shape" : (8,8,8,8),
            "in_reshaped" : None,
            "out_shape" : (8,8,8,8),
            "out_reshaped" : None,
            "perm" : (3,1,0,2)
    },
    {
            "in_shape" : (4,8,16,32), 
            "in_reshaped" : None,
            "out_shape" : (16,32,4,8),
            "out_reshaped" : None,
            "perm" : (2,3,0,1)
    },
    {
            "in_shape" : (1,256,192),
            "in_reshaped" : (1,256,6,32),
            "out_shape" : (1,6,256,32),
            "out_reshaped" : (1,6,8192),
            "perm" : (0,2,1,3)
    },
    {
            "in_shape" : (1,64,512),
            "in_reshaped" : (1,64,16,32),
            "out_shape" : (1,16,64,32),
            "out_reshaped" : None,
            "perm" : (0,2,1,3)
    },
    {
            "in_shape" : (2,32,128), 
            "in_reshaped" : (2,32,4,32),
            "out_shape" : (2,4,32,32),
            "out_reshaped" : (2,4,1024),
            "perm" : (0,2,1,3)
    },
    {
            "in_shape" : (4,4), 
            "in_reshaped" : None,
            "out_shape" : (4,4),
            "out_reshaped" : None,
            "perm" : (1,0)
    },
    {
            "in_shape" : (1,8,8), 
            "in_reshaped" : None,
            "out_shape" : (8,1,8),
            "out_reshaped" : None,
            "perm" : (1,0,2)
    },
    {
            "in_shape" : (1,1024,768),
            "in_reshaped" : (1,1024,24,32),
            "out_shape" : (1,24,1024,32),
            "out_reshaped" : None,
            "perm" : (0,2,1,3)
    },
    {
            "in_shape" : (8,128,256), 
            "in_reshaped" : None,
            "out_shape" : (256,128,8),
            "out_reshaped" : None,
            "perm" : (2,1,0)
    },
    {
            "in_shape" : (6,12,18,24),
            "in_reshaped" : None,
            "out_shape" : (18,6,24,12),
            "out_reshaped" : None,
            "perm" : (2,0,3,1)
    },
    {
            "in_shape" : (7,12,16), 
            "in_reshaped" : None,
            "out_shape" : (16,7,12),
            "out_reshaped" : None,
            "perm" : (2,0,1)
    },
    {
            "in_shape" : (5,10,15,20), 
            "in_reshaped" : None,
            "out_shape" : (15,20,5,10),
            "out_reshaped" : None,
            "perm" : (2,3,0,1)
    },
    {
            "in_shape" : (256,128), 
            "in_reshaped" : None,
            "out_shape" : (128,256),
            "out_reshaped" : None,
            "perm" : (1,0)
    },
    {
            "in_shape" : (64,96), 
            "in_reshaped" : None,
            "out_shape" : (96,64),
            "out_reshaped" : None,
            "perm" : (1,0)
    },
    {
            "in_shape" : (1,96,128), 
            "in_reshaped" : (1,96,4,32),
            "out_shape" : (1,4,96,32),
            "out_reshaped" : (1,4,3072),
            "perm" : (0,2,1,3)
    },
    {
            "in_shape" : (4,48,64), 
            "in_reshaped" : (4,48,4,16),
            "out_shape" : (4,4,48,16),
            "out_reshaped" : (4,4,768),
            "perm" : (0,2,1,3)
    },
    {
            "in_shape" : (8,32,64,16), 
            "in_reshaped" : None,
            "out_shape" : (64,8,16,32),
            "out_reshaped" : None,
            "perm" : (2,0,3,1)
    },
    {
            "in_shape" : (3,6,9,12),
            "in_reshaped" : None,
            "out_shape" : (9,12,3,6),
            "out_reshaped" : None,
            "perm" : (2,3,0,1)
    },
])
@pytest.mark.parametrize("datatype", ["INT8"])
@pytest.mark.parametrize("simd", ["simd2", "simd4"])
@pytest.mark.fpgadataflow
def test_rtlsim_shuffle_layer(shuffle_param, datatype, simd):
    ''' Checks rtlsim of the shuffle_hls layer '''
    os.environ['LIVENESS_THRESHOLD'] = '10000000' # Need to bump this up for these RTL sims
    dt = DataType[datatype]
    simd = int(simd[-1])
    in_shape = shuffle_param["in_shape"]

    model = construct_onnx_model(
            input_shape=in_shape,
            transpose_perm=shuffle_param["perm"],
            reshape1_shape=shuffle_param["in_reshaped"],
            reshape2_shape=shuffle_param["out_reshaped"],
            dt=dt
    )

    input = gen_finn_dt_tensor(dt, in_shape)
    in_name = model.graph.input[0].name
    out_name = model.graph.output[0].name
    input_t = {in_name : input}

    # Get a reference for the shuffle 
    y_ref = oxe.execute_onnx(model, input_t)[out_name]

    # Attempt to build the HLS/RTL for this
    model = model.transform(TransposeDecomposition())
    model = model.transform(InferShuffle())
    model = model.transform(SpecializeLayers(test_fpga_part))
    model = model.transform(SetShuffleSIMD(simd, enable_waveforms=True))
    model = model.transform(GiveUniqueNodeNames())
    model = model.transform(GiveReadableTensorNames())

    model = model.transform(SetExecMode("rtlsim"))
    model = model.transform(PrepareIP(test_fpga_part, test_synth_clk_period_ns))
    model = model.transform(HLSSynthIP())
    model = model.transform(PrepareRTLSim())

    y_hw = oxe.execute_onnx(model, input_t)[out_name]

    y_hw_flat = y_hw.flatten()
    y_ref_flat = y_ref.flatten()
    for i in range(len(y_hw_flat)):
        if not np.allclose(y_hw_flat[i], y_ref_flat[i]):
            print(f"Index {i}, Expected {y_ref_flat[i]} -- Got {y_hw_flat[i]}")

    assert np.allclose(y_ref, y_hw), "Model output does not match expected output"


@pytest.mark.parametrize("shuffle_param", [ 
    {
            "in_shape" : (1,128,384), # Shuffle A
            "in_reshaped" : (1,128,12,32),
            "out_shape" : (1,12,128,32),
            "out_reshaped" : None,
            "perm" : (0,2,1,3)
    }, 
    {
            "in_shape" : (1,12,128,32), # Shuffle C 
            "in_reshaped" : None,
            "out_shape" : (1,128,12,32),
            "out_reshaped" : (1,128,384),
            "perm" : (0,2,1,3)
    }, 
    {
            "in_shape" : (128,384), # pTranspose Test 
            "in_reshaped" : None,
            "out_shape" : (384,128),
            "out_reshaped" : None,
            "perm" : (1,0)
    }, 
    {
            "in_shape" : (32,16,8,12), # Mixed Transpose test 
            "in_reshaped" : None,
            "out_shape" : (8,12,32,16),
            "out_reshaped" : None,
            "perm" : (2,3,0,1)
    }, 
    {
            "in_shape" : (2,2,12,8),  
            "in_reshaped" : None,
            "out_shape" : (2,2,8,12),
            "out_reshaped" : None,
            "perm" : (0,1,3,2)
    }, 
    {
            "in_shape" : (32,16,12,8), # Mixed Transpose test 
            "in_reshaped" : None,
            "out_shape" : (8,12,16,32),
            "out_reshaped" : None,
            "perm" : (3,2,1,0)
    },
    {
            "in_shape" : (64,256), 
            "in_reshaped" : None,
            "out_shape" : (256,64),
            "out_reshaped" : None,
            "perm" : (1,0)
    },
    {
            "in_shape" : (512,128), 
            "in_reshaped" : None,
            "out_shape" : (128,512),
            "out_reshaped" : None,
            "perm" : (1,0)
    },
    {
            "in_shape" : (256,512), 
            "in_reshaped" : None,
            "out_shape" : (512,256),
            "out_reshaped" : None,
            "perm" : (1,0)
    },
    {
            "in_shape" : (8,16,32), 
            "in_reshaped" : None,
            "out_shape" : (32,16,8),
            "out_reshaped" : None,
            "perm" : (2,1,0)
    },
    {
            "in_shape" : (4,64,128), 
            "in_reshaped" : None,
            "out_shape" : (64,4,128),
            "out_reshaped" : None,
            "perm" : (1,0,2)
    },
    {
            "in_shape" : (16,8,64), 
            "in_reshaped" : None,
            "out_shape" : (64,16,8),
            "out_reshaped" : None,
            "perm" : (2,0,1)
    },
    {
            "in_shape" : (8,8,8,8),
            "in_reshaped" : None,
            "out_shape" : (8,8,8,8),
            "out_reshaped" : None,
            "perm" : (3,1,0,2)
    },
    {
            "in_shape" : (4,8,16,32), 
            "in_reshaped" : None,
            "out_shape" : (16,32,4,8),
            "out_reshaped" : None,
            "perm" : (2,3,0,1)
    },
    {
            "in_shape" : (1,256,192),
            "in_reshaped" : (1,256,6,32),
            "out_shape" : (1,6,256,32),
            "out_reshaped" : (1,6,8192),
            "perm" : (0,2,1,3)
    },
    {
            "in_shape" : (1,64,512),
            "in_reshaped" : (1,64,16,32),
            "out_shape" : (1,16,64,32),
            "out_reshaped" : None,
            "perm" : (0,2,1,3)
    },
    {
            "in_shape" : (2,32,128), 
            "in_reshaped" : (2,32,4,32),
            "out_shape" : (2,4,32,32),
            "out_reshaped" : (2,4,1024),
            "perm" : (0,2,1,3)
    },
    {
            "in_shape" : (4,4), 
            "in_reshaped" : None,
            "out_shape" : (4,4),
            "out_reshaped" : None,
            "perm" : (1,0)
    },
    {
            "in_shape" : (1,8,8), 
            "in_reshaped" : None,
            "out_shape" : (8,1,8),
            "out_reshaped" : None,
            "perm" : (1,0,2)
    },
    {
            "in_shape" : (1,1024,768),
            "in_reshaped" : (1,1024,24,32),
            "out_shape" : (1,24,1024,32),
            "out_reshaped" : None,
            "perm" : (0,2,1,3)
    },
    {
            "in_shape" : (8,128,256), 
            "in_reshaped" : None,
            "out_shape" : (256,128,8),
            "out_reshaped" : None,
            "perm" : (2,1,0)
    },
    {
            "in_shape" : (6,12,18,24),
            "in_reshaped" : None,
            "out_shape" : (18,6,24,12),
            "out_reshaped" : None,
            "perm" : (2,0,3,1)
    },
    {
            "in_shape" : (7,12,16), 
            "in_reshaped" : None,
            "out_shape" : (16,7,12),
            "out_reshaped" : None,
            "perm" : (2,0,1)
    },
    {
            "in_shape" : (5,10,15,20), 
            "in_reshaped" : None,
            "out_shape" : (15,20,5,10),
            "out_reshaped" : None,
            "perm" : (2,3,0,1)
    },
    {
            "in_shape" : (256,128), 
            "in_reshaped" : None,
            "out_shape" : (128,256),
            "out_reshaped" : None,
            "perm" : (1,0)
    },
    {
            "in_shape" : (64,96), 
            "in_reshaped" : None,
            "out_shape" : (96,64),
            "out_reshaped" : None,
            "perm" : (1,0)
    },
    {
            "in_shape" : (1,96,128), 
            "in_reshaped" : (1,96,4,32),
            "out_shape" : (1,4,96,32),
            "out_reshaped" : (1,4,3072),
            "perm" : (0,2,1,3)
    },
    {
            "in_shape" : (4,48,64), 
            "in_reshaped" : (4,48,4,16),
            "out_shape" : (4,4,48,16),
            "out_reshaped" : (4,4,768),
            "perm" : (0,2,1,3)
    },
    {
            "in_shape" : (8,32,64,16), 
            "in_reshaped" : None,
            "out_shape" : (64,8,16,32),
            "out_reshaped" : None,
            "perm" : (2,0,3,1)
    },
    {
            "in_shape" : (3,6,9,12),
            "in_reshaped" : None,
            "out_shape" : (9,12,3,6),
            "out_reshaped" : None,
            "perm" : (2,3,0,1)
    },
])
@pytest.mark.parametrize("datatype", ["INT8"])
@pytest.mark.parametrize("simd", ["simd2", "simd4"])
@pytest.mark.fpgadataflow
@pytest.mark.vivado
@pytest.mark.slow
def test_stitched_ip_shuffle_layer(shuffle_param, datatype, simd):
    ''' Build stitched IP for shuffle layer tests and save results for buffer analysis '''
    dt = DataType[datatype]
    simd = int(simd[-1])
    in_shape = shuffle_param["in_shape"]

    model = construct_onnx_model(
            input_shape=in_shape,
            transpose_perm=shuffle_param["perm"],
            reshape1_shape=shuffle_param["in_reshaped"],
            reshape2_shape=shuffle_param["out_reshaped"],
            dt=dt
    )

    model = model.transform(TransposeDecomposition())
    model = model.transform(InferShuffle())
    model = model.transform(SpecializeLayers(test_fpga_part))
    model = model.transform(SetShuffleSIMD(simd))
    model = model.transform(GiveUniqueNodeNames())
    model = model.transform(GiveReadableTensorNames())

    model = model.transform(PrepareIP(test_fpga_part, test_synth_clk_period_ns))
    model = model.transform(HLSSynthIP())
    
    model = model.transform(CreateStitchedIP(test_fpga_part, test_synth_clk_period_ns))
    
    model = model.transform(SynthOutOfContext(test_fpga_part, test_synth_clk_period_ns))
    
    results_base_dir = "./shuffle_stitched_ip_analysis"
    os.makedirs(results_base_dir, exist_ok=True)
    
    param_str = f"{datatype}_simd{simd}_{hash(str(shuffle_param['in_shape']) + str(shuffle_param['perm']))}"
    results_dir = os.path.join(results_base_dir, param_str)
    os.makedirs(results_dir, exist_ok=True)
    
    model.save(os.path.join(results_dir, "stitched_model.onnx"))
    
    vivado_stitch_proj_dir = model.get_metadata_prop("vivado_stitch_proj")
    if vivado_stitch_proj_dir and os.path.isdir(vivado_stitch_proj_dir):
        import shutil
        target_proj_dir = os.path.join(results_dir, "vivado_stitch_proj")
        shutil.copytree(vivado_stitch_proj_dir, target_proj_dir, dirs_exist_ok=True)
        
        # Save test parameters and synthesis results for reference
        import json
        param_info = {
            "shuffle_param": shuffle_param,
            "datatype": datatype,
            "simd": simd,
            "vivado_proj_path": target_proj_dir,
            "original_proj_path": vivado_stitch_proj_dir,
            "synthesis_results": model.get_metadata_prop("res_total_ooc_synth")
        }
        with open(os.path.join(results_dir, "test_params.json"), "w") as f:
            json.dump(param_info, f, indent=2)
        
        print(f"Stitched IP results saved to: {results_dir}")
        
        synth_results = model.get_metadata_prop("res_total_ooc_synth")
        if synth_results:
            results_dict = eval(synth_results)
            print(f"Resource usage - LUT: {results_dict.get('LUT', 'N/A')}, FF: {results_dict.get('FF', 'N/A')}, BRAM: {results_dict.get('BRAM', 'N/A')}, DSP: {results_dict.get('DSP', 'N/A')}")
    
    assert vivado_stitch_proj_dir is not None, "Stitched IP project was not created"
    assert os.path.isdir(vivado_stitch_proj_dir), "Stitched IP project directory does not exist"
    

