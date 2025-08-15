# Copyright (C) 2025, Advanced Micro Devices, Inc.
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# * Redistributions of source code must retain the above copyright notice, this
#   list of conditions and the following disclaimer.
#
# * Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.
#
# * Neither the name of FINN nor the names of its
#   contributors may be used to endorse or promote products derived from
#   this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

import pytest

import numpy as np
import tempfile
import torch
from brevitas.export import export_qonnx
from brevitas.nn import QuantEltwiseAdd
from onnx import TensorProto
from onnx import helper as oh
from qonnx.core.datatype import DataType
from qonnx.core.modelwrapper import ModelWrapper
from qonnx.core.onnx_exec import execute_onnx
from qonnx.transformation.general import (
    ApplyConfig,
    GiveReadableTensorNames,
    GiveUniqueNodeNames,
    GiveUniqueParameterTensors,
    RemoveUnusedTensors,
)
from qonnx.transformation.infer_data_layouts import InferDataLayouts
from qonnx.transformation.infer_datatypes import InferDataTypes
from qonnx.transformation.infer_shapes import InferShapes
from qonnx.util.basic import gen_finn_dt_tensor, qonnx_make_model

from finn.transformation.fpgadataflow.compile_cppsim import CompileCppSim
from finn.transformation.fpgadataflow.convert_to_hw_layers import (
    InferElementwiseBinaryOperation,
    InferThresholdingLayer,
)
from finn.transformation.fpgadataflow.hlssynth_ip import HLSSynthIP
from finn.transformation.fpgadataflow.minimize_accumulator_width import (
    MinimizeAccumulatorWidth,
)
from finn.transformation.fpgadataflow.minimize_weight_bit_width import (
    MinimizeWeightBitWidth,
)
from finn.transformation.fpgadataflow.prepare_cppsim import PrepareCppSim
from finn.transformation.fpgadataflow.prepare_ip import PrepareIP
from finn.transformation.fpgadataflow.prepare_rtlsim import PrepareRTLSim
from finn.transformation.fpgadataflow.set_exec_mode import SetExecMode
from finn.transformation.fpgadataflow.specialize_layers import SpecializeLayers
from finn.transformation.qonnx.convert_qonnx_to_finn import ConvertQONNXtoFINN
from finn.transformation.streamline import Streamline
from finn.transformation.streamline.absorb import (
    AbsorbMulIntoMultiThreshold,
    AbsorbSignBiasIntoMultiThreshold,
)
from finn.transformation.streamline.reorder import MoveMulPastJoinAdd

# Mapping of ElementwiseBinaryOperation specializations to numpy reference
# implementation functions
NUMPY_REFERENCES = {
    "ElementwiseAdd": np.add,
    "ElementwiseSub": np.subtract,
    "ElementwiseMul": np.multiply,
    # TODO: "ElementwiseDiv": np.divide, Cannot guarantee non-zero test input
    # TODO: "ElementwiseMod": np.mode / np.fmod
    "ElementwiseAnd": np.logical_and,
    "ElementwiseOr": np.logical_or,
    "ElementwiseXor": np.logical_xor,
    "ElementwiseEqual": np.equal,
    "ElementwiseLess": np.less,
    "ElementwiseLessOrEqual": np.less_equal,
    "ElementwiseGreater": np.greater,
    "ElementwiseGreaterOrEqual": np.greater_equal,
    "ElementwiseBitwiseAnd": np.bitwise_and,
    "ElementwiseBitwiseOr": np.bitwise_or,
    "ElementwiseBitwiseXor": np.bitwise_xor,
    # TODO: "ElementwiseBitShift": np.left_shift / np.right_shift
    # TODO: "ElementwisePow": np.power
}

# Names of bitwise operations which somtimes require special treatment
BITWISE = ["ElementwiseBitwiseAnd", "ElementwiseBitwiseOr", "ElementwiseBitwiseXor"]


# Creates a model executing a binary elementwise operation
def mock_elementwise_binary_operation(
    op_type, lhs_dtype, rhs_dtype, out_dtype, lhs_shape, rhs_shape, pe
):
    # Automatically derive the output shape by broadcasting the inputs
    out_shape = np.broadcast_shapes(lhs_shape, rhs_shape)
    # Create a node representing the binary elementwise operation
    node = oh.make_node(
        # Operator type from the name of the fpgadataflow hlscustomop
        op_type=op_type,
        # Specify the domain, i.e., the package to look for the custom operator
        # implementation
        domain="finn.custom_op.fpgadataflow",
        # Execution backend: Required attribute inherited from HLSCustomOp
        backend="fpgadataflow",
        # Just one input
        inputs=["lhs", "rhs"],
        # Enumerate the outputs
        outputs=["out"],
        # Data type of the left-hand-side input elements
        lhs_dtype=lhs_dtype,
        # Data type of the right-hand-side input elements
        rhs_dtype=rhs_dtype,
        # Data type of the output elements
        out_dtype=out_dtype,
        # Shape of the left-hand-side input
        lhs_shape=lhs_shape,
        # Shape of the right-hand-side input
        rhs_shape=rhs_shape,
        # Shape of the output, mus correspond to multi-directional
        # broadcasting of the left- and right-hand-side
        out_shape=out_shape,
        # Number of elements to process in parallel
        PE=pe,
    )
    # Construct the input tensor value infos
    lhs = oh.make_tensor_value_info("lhs", TensorProto.FLOAT, lhs_shape)
    rhs = oh.make_tensor_value_info("rhs", TensorProto.FLOAT, rhs_shape)
    # Construct output tensor value infos
    out = oh.make_tensor_value_info("out", TensorProto.FLOAT, out_shape)
    # Create a graph connecting the node to the inputs and outputs
    graph = oh.make_graph([node], inputs=[lhs, rhs], outputs=[out], name="elementwise-binary")
    # Wrap the ONNX graph in QONNX model wrapper
    model = ModelWrapper(qonnx_make_model(graph, producer_name="elementwise-binary"))

    # Add datatype annotation to the value info of input tensors
    model.set_tensor_datatype("lhs", DataType[lhs_dtype])
    model.set_tensor_datatype("rhs", DataType[rhs_dtype])
    model.set_tensor_datatype("out", DataType[out_dtype])

    # Return the wrapped onnx model
    return model


# Operator type to be tested
@pytest.mark.parametrize(
    "op_type",
    [
        # Test all Numpy references specified above
        *NUMPY_REFERENCES.keys()
    ],
)
# Data type of the left-hand-side input elements
@pytest.mark.parametrize("lhs_dtype", ["INT8"])
# Data type of the right-hand-side input elements
@pytest.mark.parametrize("rhs_dtype", ["INT8"])
# Data type of the output elements
@pytest.mark.parametrize("out_dtype", ["INT32"])
# Shape of the left-hand-side input
@pytest.mark.parametrize("lhs_shape", [[3, 1, 7, 1], [1]])
# Shape of the right-hand-side input
@pytest.mark.parametrize(
    "rhs_shape",
    [
        [3, 32, 1, 16],
    ],
)
# Which inputs to set as initializers
@pytest.mark.parametrize("initializers", [[], ["lhs"], ["rhs"], ["lhs", "rhs"]])
# Number of elements to process in parallel
@pytest.mark.parametrize("pe", [1, 2, 4])
def test_elementwise_binary_operation_python(
    op_type, lhs_dtype, rhs_dtype, out_dtype, lhs_shape, rhs_shape, pe, initializers
):
    # Make dummy model for testing
    model = mock_elementwise_binary_operation(
        op_type, lhs_dtype, rhs_dtype, out_dtype, lhs_shape, rhs_shape, pe
    )
    # Prepare the execution context
    context = {
        "lhs": gen_finn_dt_tensor(DataType[lhs_dtype], lhs_shape),
        "rhs": gen_finn_dt_tensor(DataType[rhs_dtype], rhs_shape),
    }

    # Turn selected inputs into initializers
    for name in initializers:
        model.set_initializer(name, context[name])

    # Get the numpy reference implementation for this operation
    numpy_reference = NUMPY_REFERENCES[op_type]

    # Test running shape and data type inference on the model graph
    model = model.transform(InferDataTypes())
    model = model.transform(InferShapes())

    # Try to minimize the bit-widths of all data types involved
    model = model.transform(MinimizeWeightBitWidth())
    model = model.transform(MinimizeAccumulatorWidth())

    # Set model execution mode to python simulation
    model = model.transform(SetExecMode("python"))
    model = model.transform(GiveUniqueNodeNames())

    # Compute ground-truth output in software
    o_expected = numpy_reference(
        # Note: Need to make sure these have the right type for the Numpy API
        # Note: Assume all test cases fit into int64 without loss of precision
        context["lhs"].astype(np.int64),
        context["rhs"].astype(np.int64),
    )
    # Execute the onnx model to collect the result
    o_produced = execute_onnx(model, context)["out"]

    # Compare the expected to the produced for exact equality
    assert np.all(o_produced == o_expected)


# Operator type to be tested
@pytest.mark.parametrize(
    "op_type",
    [
        # Test all Numpy references specified above, except for the bitwise
        # operations, for which floating-point does not make sense
        *sorted((NUMPY_REFERENCES.keys() - BITWISE)),
    ],
)
# Data type of the left-hand-side input elements
@pytest.mark.parametrize("lhs_dtype", ["FLOAT32"])
# Data type of the right-hand-side input elements
@pytest.mark.parametrize("rhs_dtype", ["FLOAT32"])
# Data type of the output elements
@pytest.mark.parametrize("out_dtype", ["FLOAT32"])
# Shape of the left-hand-side input
@pytest.mark.parametrize("lhs_shape", [[3, 1, 7, 1], [1]])
# Shape of the right-hand-side input
@pytest.mark.parametrize(
    "rhs_shape",
    [
        [3, 32, 1, 16],
    ],
)
# Which inputs to set as initializers
@pytest.mark.parametrize("initializers", [[], ["lhs"], ["rhs"], ["lhs", "rhs"]])
# Number of elements to process in parallel
@pytest.mark.parametrize("pe", [1, 2, 4])
def test_elementwise_binary_operation_float_python(
    op_type, lhs_dtype, rhs_dtype, out_dtype, lhs_shape, rhs_shape, pe, initializers
):
    # Make dummy model for testing
    model = mock_elementwise_binary_operation(
        op_type, lhs_dtype, rhs_dtype, out_dtype, lhs_shape, rhs_shape, pe
    )
    # Prepare the execution context
    context = {
        "lhs": gen_finn_dt_tensor(DataType[lhs_dtype], lhs_shape),
        "rhs": gen_finn_dt_tensor(DataType[rhs_dtype], rhs_shape),
    }

    # Turn selected inputs into initializers
    for name in initializers:
        model.set_initializer(name, context[name])

    # Get the numpy reference implementation for this operation
    numpy_reference = NUMPY_REFERENCES[op_type]

    # Test running shape and data type inference on the model graph
    model = model.transform(InferDataTypes())
    model = model.transform(InferShapes())

    # Try to minimize the bit-widths of all data types involved
    model = model.transform(MinimizeWeightBitWidth())
    model = model.transform(MinimizeAccumulatorWidth())

    # Set model execution mode to python simulation
    model = model.transform(SetExecMode("python"))
    model = model.transform(GiveUniqueNodeNames())

    # Compute ground-truth output in software
    o_expected = numpy_reference(context["lhs"], context["rhs"])
    # Execute the onnx model to collect the result
    o_produced = execute_onnx(model, context)["out"]

    # Compare the expected to the produced for exact equality
    assert np.all(o_produced == o_expected)


# Operator type to be tested
@pytest.mark.parametrize(
    "op_type",
    [
        # Test all Numpy references specified above
        *NUMPY_REFERENCES.keys(),
    ],
)
# Data type of the left-hand-side input elements
@pytest.mark.parametrize("lhs_dtype", ["INT8"])
# Data type of the right-hand-side input elements
@pytest.mark.parametrize("rhs_dtype", ["INT8"])
# Data type of the output elements
@pytest.mark.parametrize("out_dtype", ["INT32"])
# Shape of the left-hand-side input
@pytest.mark.parametrize("lhs_shape", [[3, 1, 7, 1], [1]])
# Shape of the right-hand-side input
@pytest.mark.parametrize(
    "rhs_shape",
    [
        [3, 32, 1, 16],
    ],
)
# Which inputs to set as initializers
@pytest.mark.parametrize("initializers", [[], ["lhs"], ["rhs"], ["lhs", "rhs"]])
# Number of elements to process in parallel
@pytest.mark.parametrize("pe", [1, 2, 4])
# This is a slow running fpgadataflow type of test which requires vivado
@pytest.mark.fpgadataflow
@pytest.mark.slow
def test_elementwise_binary_operation_cppsim(
    op_type, lhs_dtype, rhs_dtype, out_dtype, lhs_shape, rhs_shape, pe, initializers
):
    # Make dummy model for testing
    model = mock_elementwise_binary_operation(
        op_type, lhs_dtype, rhs_dtype, out_dtype, lhs_shape, rhs_shape, pe
    )
    # Prepare the execution context
    context = {
        "lhs": gen_finn_dt_tensor(DataType[lhs_dtype], lhs_shape),
        "rhs": gen_finn_dt_tensor(DataType[rhs_dtype], rhs_shape),
    }

    # Turn selected inputs into initializers
    for name in initializers:
        model.set_initializer(name, context[name])

    # Get the numpy reference implementation for this operation
    numpy_reference = NUMPY_REFERENCES[op_type]

    # Test running shape and data type inference on the model graph
    model = model.transform(InferDataTypes())
    model = model.transform(InferShapes())
    # Specializes all nodes to be implemented as HLS backend
    model = model.transform(SpecializeLayers("xczu7ev-ffvc1156-2-e"))

    assert len(model.graph.node) == 1
    assert model.graph.node[0].op_type == f"{op_type}_hls"

    # Try to minimize the bit-widths of all data types involved
    model = model.transform(MinimizeWeightBitWidth())
    model = model.transform(MinimizeAccumulatorWidth())

    # Set model execution mode to C++ simulation
    model = model.transform(SetExecMode("cppsim"))
    # Generates the C++ source and compiles the C++ simulation
    model = model.transform(GiveUniqueNodeNames())
    model = model.transform(PrepareCppSim())
    model = model.transform(CompileCppSim())

    # Compute ground-truth output in software
    o_expected = numpy_reference(
        # Note: Need to make sure these have the right type for the Numpy API
        # Note: Assume all test cases fit into int64 without loss of precision
        context["lhs"].astype(np.int64),
        context["rhs"].astype(np.int64),
    )
    # Execute the onnx model to collect the result
    o_produced = execute_onnx(model, context)["out"]

    # Compare the expected to the produced for exact equality
    assert np.all(o_produced == o_expected)


# Operator type to be tested
@pytest.mark.parametrize(
    "op_type",
    [
        # Test all Numpy references specified above, except for the bitwise
        # operations, for which floating-point does not make sense
        *sorted((NUMPY_REFERENCES.keys() - BITWISE)),
    ],
)
# Data type of the left-hand-side input elements
@pytest.mark.parametrize("lhs_dtype", ["FLOAT32"])
# Data type of the right-hand-side input elements
@pytest.mark.parametrize("rhs_dtype", ["FLOAT32"])
# Data type of the output elements
@pytest.mark.parametrize("out_dtype", ["FLOAT32"])
# Shape of the left-hand-side input
@pytest.mark.parametrize("lhs_shape", [[3, 1, 7, 1], [1]])
# Shape of the right-hand-side input
@pytest.mark.parametrize(
    "rhs_shape",
    [
        [3, 32, 1, 16],
    ],
)
# Which inputs to set as initializers
@pytest.mark.parametrize("initializers", [[], ["lhs"], ["rhs"], ["lhs", "rhs"]])
# Number of elements to process in parallel
@pytest.mark.parametrize("pe", [1, 2, 4])
# This is a slow running fpgadataflow type of test which requires vivado
@pytest.mark.fpgadataflow
@pytest.mark.slow
def test_elementwise_binary_operation_float_cppsim(
    op_type, lhs_dtype, rhs_dtype, out_dtype, lhs_shape, rhs_shape, pe, initializers
):
    # Make dummy model for testing
    model = mock_elementwise_binary_operation(
        op_type, lhs_dtype, rhs_dtype, out_dtype, lhs_shape, rhs_shape, pe
    )
    # Prepare the execution context
    context = {
        "lhs": gen_finn_dt_tensor(DataType[lhs_dtype], lhs_shape),
        "rhs": gen_finn_dt_tensor(DataType[rhs_dtype], rhs_shape),
    }

    # Turn selected inputs into initializers
    for name in initializers:
        model.set_initializer(name, context[name])

    # Get the numpy reference implementation for this operation
    numpy_reference = NUMPY_REFERENCES[op_type]

    # Test running shape and data type inference on the model graph
    model = model.transform(InferDataTypes())
    model = model.transform(InferShapes())
    # Specializes all nodes to be implemented as HLS backend
    model = model.transform(SpecializeLayers("xczu7ev-ffvc1156-2-e"))

    assert len(model.graph.node) == 1
    assert model.graph.node[0].op_type == f"{op_type}_hls"

    # Try to minimize the bit-widths of all data types involved
    model = model.transform(MinimizeWeightBitWidth())
    model = model.transform(MinimizeAccumulatorWidth())

    # Set model execution mode to C++ simulation
    model = model.transform(SetExecMode("cppsim"))
    # Generates the C++ source and compiles the C++ simulation
    model = model.transform(GiveUniqueNodeNames())
    model = model.transform(PrepareCppSim())
    model = model.transform(CompileCppSim())

    # Compute ground-truth output in software
    o_expected = numpy_reference(context["lhs"], context["rhs"])
    # Execute the onnx model to collect the result
    o_produced = execute_onnx(model, context)["out"]

    # Compare the expected to the produced for exact equality
    assert np.all(o_produced == o_expected)


# Operator type to be tested
@pytest.mark.parametrize(
    "op_type",
    [
        # Test all Numpy references specified above
        *NUMPY_REFERENCES.keys()
    ],
)
# Data type of the left-hand-side input elements
@pytest.mark.parametrize("lhs_dtype", ["INT8"])
# Data type of the right-hand-side input elements
@pytest.mark.parametrize("rhs_dtype", ["INT8"])
# Data type of the output elements
@pytest.mark.parametrize("out_dtype", ["INT32"])
# Shape of the left-hand-side input
@pytest.mark.parametrize("lhs_shape", [[3, 1, 7, 1], [1]])
# Shape of the right-hand-side input
@pytest.mark.parametrize(
    "rhs_shape",
    [
        [3, 32, 1, 16],
    ],
)
# Which inputs to set as initializers
@pytest.mark.parametrize("initializers", [[], ["lhs"], ["rhs"], ["lhs", "rhs"]])
# Number of elements to process in parallel
@pytest.mark.parametrize("pe", [1, 2, 4])
# This is a slow running fpgadataflow type of test which requires vivado
@pytest.mark.fpgadataflow
@pytest.mark.slow
def test_elementwise_binary_operation_rtlsim(
    op_type, lhs_dtype, rhs_dtype, out_dtype, lhs_shape, rhs_shape, pe, initializers
):
    # Make dummy model for testing
    model = mock_elementwise_binary_operation(
        op_type, lhs_dtype, rhs_dtype, out_dtype, lhs_shape, rhs_shape, pe
    )
    # Prepare the execution context
    context = {
        "lhs": gen_finn_dt_tensor(DataType[lhs_dtype], lhs_shape),
        "rhs": gen_finn_dt_tensor(DataType[rhs_dtype], rhs_shape),
    }

    # Turn selected inputs into initializers
    for name in initializers:
        model.set_initializer(name, context[name])

    # Get the numpy reference implementation for this operation
    numpy_reference = NUMPY_REFERENCES[op_type]

    # Test running shape and data type inference on the model graph
    model = model.transform(InferDataTypes())
    model = model.transform(InferShapes())
    # Specializes all nodes to be implemented as HLS backend
    model = model.transform(SpecializeLayers("xczu7ev-ffvc1156-2-e"))

    assert len(model.graph.node) == 1
    assert model.graph.node[0].op_type == f"{op_type}_hls"

    # Try to minimize the bit-widths of all data types involved
    model = model.transform(MinimizeWeightBitWidth())
    model = model.transform(MinimizeAccumulatorWidth())

    # Set model execution mode to RTL simulation
    model = model.transform(SetExecMode("rtlsim"))
    # Generates the C++ source and compiles the RTL simulation
    model = model.transform(GiveUniqueNodeNames())
    model = model.transform(PrepareIP("xczu7ev-ffvc1156-2-e", 10))
    model = model.transform(HLSSynthIP())
    model = model.transform(PrepareRTLSim())

    # Compute ground-truth output in software
    o_expected = numpy_reference(
        # Note: Need to make sure these have the right type for the Numpy API
        # Note: Assume all test cases fit into int64 without loss of precision
        context["lhs"].astype(np.int64),
        context["rhs"].astype(np.int64),
    )
    # Execute the onnx model to collect the result
    o_produced = execute_onnx(model, context)["out"]

    # Compare the expected to the produced for exact equality
    assert np.all(o_produced == o_expected)


# Operator type to be tested
@pytest.mark.parametrize(
    "op_type",
    [
        # Test all Numpy references specified above, except for the bitwise
        # operations, for which floating-point does not make sense
        *sorted((NUMPY_REFERENCES.keys() - BITWISE)),
    ],
)
# Data type of the left-hand-side input elements
@pytest.mark.parametrize("lhs_dtype", ["FLOAT32"])
# Data type of the right-hand-side input elements
@pytest.mark.parametrize("rhs_dtype", ["FLOAT32"])
# Data type of the output elements
@pytest.mark.parametrize("out_dtype", ["FLOAT32"])
# Shape of the left-hand-side input
@pytest.mark.parametrize("lhs_shape", [[3, 1, 7, 1], [1]])
# Shape of the right-hand-side input
@pytest.mark.parametrize(
    "rhs_shape",
    [
        [3, 32, 1, 16],
    ],
)
# Which inputs to set as initializers
@pytest.mark.parametrize("initializers", [[], ["lhs"], ["rhs"], ["lhs", "rhs"]])
# Number of elements to process in parallel
@pytest.mark.parametrize("pe", [1, 2, 4])
# This is a slow running fpgadataflow type of test which requires vivado
@pytest.mark.fpgadataflow
@pytest.mark.slow
def test_elementwise_binary_operation_float_rtlsim(
    op_type, lhs_dtype, rhs_dtype, out_dtype, lhs_shape, rhs_shape, pe, initializers
):
    # Make dummy model for testing
    model = mock_elementwise_binary_operation(
        op_type, lhs_dtype, rhs_dtype, out_dtype, lhs_shape, rhs_shape, pe
    )
    # Prepare the execution context
    context = {
        "lhs": gen_finn_dt_tensor(DataType[lhs_dtype], lhs_shape),
        "rhs": gen_finn_dt_tensor(DataType[rhs_dtype], rhs_shape),
    }

    # Turn selected inputs into initializers
    for name in initializers:
        model.set_initializer(name, context[name])

    # Get the numpy reference implementation for this operation
    numpy_reference = NUMPY_REFERENCES[op_type]

    # Test running shape and data type inference on the model graph
    model = model.transform(InferDataTypes())
    model = model.transform(InferShapes())
    # Specializes all nodes to be implemented as HLS backend
    model = model.transform(SpecializeLayers("xczu7ev-ffvc1156-2-e"))

    assert len(model.graph.node) == 1
    assert model.graph.node[0].op_type == f"{op_type}_hls"

    # Try to minimize the bit-widths of all data types involved
    model = model.transform(MinimizeWeightBitWidth())
    model = model.transform(MinimizeAccumulatorWidth())

    # Set model execution mode to RTL simulation
    model = model.transform(SetExecMode("rtlsim"))
    # Generates the C++ source and compiles the RTL simulation
    model = model.transform(GiveUniqueNodeNames())
    model = model.transform(PrepareIP("xczu7ev-ffvc1156-2-e", 10))
    model = model.transform(HLSSynthIP())
    model = model.transform(PrepareRTLSim())

    # Compute ground-truth output in software
    o_expected = numpy_reference(context["lhs"], context["rhs"])
    # Execute the onnx model to collect the result
    o_produced = execute_onnx(model, context)["out"]

    # Compare the expected to the produced for exact equality
    assert np.all(o_produced == o_expected)


# Test-case setting up a complete dummy model containing various elementwise
# binary operations in PyTorch, converting to QONNX and verifying in Python, C++
# and RTL simulation
# Shape of the left-hand-side input
# Note: Stripped down test of broadcasting semantics due to rather poor support
# for arbitrary data layouts inf QONNX and FINN: Only 2d and 4d layouts (with
# certain assumptions/restrictions) are really supported.
# Note: Cannot test scalar shapes (or effectively scalar shapes like [1,1]), due
# to streamlining integrating those into MultiThresholds (removing the operator
# to be tested), leading to consecutive quantizers. Consecutive quantizers
# should be avoided as  this sometimes can cause range and precision errors.
@pytest.mark.parametrize("lhs_shape", [[32, 1]])
# Shape of the right-hand-side input
@pytest.mark.parametrize("rhs_shape", [[32, 16]])
# Which inputs to set as initializers
@pytest.mark.parametrize("initializers", [[], ["lhs"], ["rhs"]])
# Number of elements to process in parallel
@pytest.mark.parametrize("pe", [1, 2, 4])
# This is a slow running fpgadataflow type of test which requires vivado
@pytest.mark.fpgadataflow
@pytest.mark.slow
def test_elementwise_binary_operation_integration_elementwise_add(
    lhs_shape, rhs_shape, initializers, pe
):
    # PyTorch model wrapping the component(s) to be tested
    class Dummy(torch.nn.Module):
        # Sets up the test model and initializes parameters
        def __init__(self):
            # Initialize the PyTorch Module superclass
            super().__init__()
            # Elementwise addition component to be tested
            self.add = QuantEltwiseAdd()
            # Left- and right-hand-side input tensors in case these are set to
            # be initializers
            self.lhs = torch.randn(*lhs_shape)
            self.rhs = torch.randn(*rhs_shape)

        # Model forward pass taking multiple inputs as arguments
        def forward(self, *xs):
            # Depending on the test configuration, extract inputs to the add
            # operation from model inputs of from model parameters
            _lhs = self.lhs if "lhs" in initializers else xs[0]
            _rhs = self.rhs if "rhs" in initializers else xs[1]
            # Quantized elementwise addition of the two inputs
            return self.add(_lhs, _rhs)

    # Create the test instance of the dummy model
    model = Dummy()
    # Create dummy test inputs
    lhs = torch.randn(*lhs_shape)
    rhs = torch.randn(*rhs_shape)
    # Do a forward pass with model in training mode to calibrate the quantizers
    _ = model(lhs, rhs)
    # Switch model to evaluation mode to keep parameters fixed for export
    model = model.eval()
    # Do not accumulate gradients while generating test output
    with torch.no_grad():
        # Model forward pass generating the expected output for verification
        out_expected = model(lhs, rhs).numpy().astype(np.float32)
    # Generate a temporary directory for running this test
    with tempfile.TemporaryDirectory() as tmp:
        # Export the model to ONNX format to be consumed by FINN
        export_qonnx(model, (lhs, rhs), tmp + "/model.onnx")
        # Wrap the model with QONNX wrapper for transformations
        model = ModelWrapper(tmp + "/model.onnx")
        # Cleanup transformations preparing the model to be consumed by FINN
        model = model.transform(InferDataTypes())
        model = model.transform(InferShapes())
        model = model.transform(InferDataLayouts())
        model = model.transform(ConvertQONNXtoFINN())
        model = model.transform(GiveUniqueNodeNames())
        model = model.transform(GiveUniqueParameterTensors())
        model = model.transform(GiveReadableTensorNames())
        model = model.transform(RemoveUnusedTensors())
        # Need to absorb scalar multiplication into the thresholding layer
        # first, to prevent large rounding error due to moving these in front of
        # add operations later.
        model = model.transform(AbsorbMulIntoMultiThreshold())
        # Need to absorb the sign bias of the quantizer back into the
        # corresponding thresholds first instead of moving them past the next
        # operator to avoid sign and range issues.
        model = model.transform(AbsorbSignBiasIntoMultiThreshold())
        # There might be identical Mul in front of the joining Add node
        model = model.transform(MoveMulPastJoinAdd())
        model = model.transform(AbsorbMulIntoMultiThreshold())
        # Do a single round of standard streamlining of the model graph
        model = model.transform(Streamline())
        # Convert layers to hardware custom operations
        model = model.transform(InferThresholdingLayer())
        model = model.transform(
            InferElementwiseBinaryOperation(
                # We want to keep the output de-quantization off-chip
                _filter=InferElementwiseBinaryOperation.reject_floats
            )
        )

        # Apply folding config to set the PE parallelism for hardware layers
        model = model.transform(
            ApplyConfig({"Defaults": {"PE": [pe, ["ElementwiseAdd", "Thresholding"]]}})
        )

        # Try to minimize the bit-widths of all data types involved
        model = model.transform(MinimizeWeightBitWidth())
        model = model.transform(MinimizeAccumulatorWidth())

        # Prepare the execution context with dummy data from above and input
        # node names extracted from transformed modelo graph
        context = {}

        # Convert verification inputs to numpy format used by ONNX execution
        lhs = lhs.numpy().astype(np.float32)
        rhs = rhs.numpy().astype(np.float32)

        # If the left-hand-side is not an initializer, it must be an input
        # inserted into the execution context
        if "lhs" not in initializers:
            # Left-hand-side is always the first input
            context[model.graph.input[0].name] = lhs

        # If the right-hand-side is not an initializer, it must be an input
        # inserted into the execution context
        if "rhs" not in initializers:
            # Index of the right-hand-side input depends on whether there is a
            # left-hand-side input
            rhs_index = int("lhs" not in initializers)
            context[model.graph.input[rhs_index].name] = rhs

        # Set model execution mode to python simulation
        model = model.transform(SetExecMode("python"))
        model = model.transform(GiveUniqueNodeNames())
        # Execute the onnx model to collect the result
        out_produced = execute_onnx(model, context)[model.graph.output[0].name]
        # Compare the expected to the produced
        # Note: Only test for close up to some tolerance as the modelo has
        # streamlined, which may involve rounding
        assert np.allclose(
            out_produced, out_expected, atol=1e-3
        ), "Python simulation verification failed"

        # Apply folding config to implement Thresholding layers in RTL mode
        # Note: Must be done in RTL for now to avoid test failing due to
        # PE-parallel stream being too wide for Vitis HLS.
        model = model.transform(
            ApplyConfig({"Defaults": {"preferred_impl_style": ["rtl", ["Thresholding"]]}})
        )
        # # Specializes all nodes to their backend implementation
        model = model.transform(SpecializeLayers("xczu7ev-ffvc1156-2-e"))

        # Set model execution mode to C++ simulation
        model = model.transform(SetExecMode("cppsim"))
        model = model.transform(GiveUniqueNodeNames())
        # Generates the C++ source and compiles the C++ simulation
        model = model.transform(PrepareCppSim())
        model = model.transform(CompileCppSim())
        # Execute the onnx model to collect the result
        out_produced = execute_onnx(model, context)[model.graph.output[0].name]
        # Compare the expected to the produced
        # Note: Only test for close up to some tolerance as the modelo has
        # streamlined, which may involve rounding
        assert np.allclose(
            out_produced, out_expected, atol=1e-3
        ), "C++ simulation verification failed"

        # Set model execution mode to RTL simulation
        model = model.transform(SetExecMode("rtlsim"))
        model = model.transform(GiveUniqueNodeNames())
        # Generates the C++ source and compiles the RTL simulation
        model = model.transform(PrepareIP("xczu7ev-ffvc1156-2-e", 10))
        model = model.transform(HLSSynthIP())
        model = model.transform(PrepareRTLSim())
        # Execute the onnx model to collect the result
        out_produced = execute_onnx(model, context)[model.graph.output[0].name]
        # Compare the expected to the produced
        # Note: Only test for close up to some tolerance as the modelo has
        # streamlined, which may involve rounding
        assert np.allclose(
            out_produced, out_expected, atol=1e-3
        ), "RTL simulation verification failed"
