############################################################################
# Copyright (C) 2025, Advanced Micro Devices, Inc.
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# @author       Shane T. Fleming <shane.fleming@amd.com>
############################################################################
import os
import shutil
import numpy as np
import warnings
from onnx.helper import make_node
from qonnx.core.datatype import DataType
from finn.custom_op.fpgadataflow.ptranspose import PTranspose
from finn.custom_op.fpgadataflow.rtlbackend import RTLBackend


class PTranspose_rtl(PTranspose, RTLBackend):
    """ CustomOp wrapper for the finn-rtllib ptranspose component. """

    def __init__(self, onnx_node, **kwargs):
        super().__init__(onnx_node, **kwargs)

    def get_nodeattr_types(self):
        my_attrs = {}
        my_attrs.update(PTranspose.get_nodeattr_types(self))
        my_attrs.update(RTLBackend.get_nodeattr_types(self))
        return my_attrs

    def get_template_values(self, idims, simd, dt):
        code_gen_dict = {
            "TOP_MODULE_NAME" : self.get_verilog_top_module_name(),
            "I" : idims[0],
            "J" : idims[1],
            "SIMD" : simd,
            "WIDTH" : dt.bitwidth(),
            "STREAM_BITS" : simd * dt.bitwidth()
        }
        return code_gen_dict

    def generate_hdl(self, model, fpgapart, clk):
        rtlsrc = f'{os.environ["FINN_ROOT"]}/finn-rtllib/ptranspose'
        template_path = f"{rtlsrc}/ptranspose_template.v"
        code_gen_dir = self.get_nodeattr("code_gen_dir_ipgen")
        dt = DataType[self.get_nodeattr("data_type")] 
        simd = self.get_nodeattr("SIMD") 
        code_gen_dict = {
            "TOP_MODULE_NAME" : self.get_verilog_top_module_name(),
            "I" : self.get_nodeattr("in_shape")[0],
            "J" : self.get_nodeattr("in_shape")[1],
            "SIMD" : simd,
            "WIDTH" : dt.bitwidth(),
            "STREAM_BITS" : simd * dt.bitwidth()
        }
        with open(template_path, "r") as f:
            template = f.read()
        for key_name in code_gen_dict:
            key = f"${key_name}$"
            template = template.replace(key, str(code_gen_dict[key_name]))

        with open(
                os.path.join(code_gen_dir, f"{self.get_verilog_top_module_name()}.v"),
                "w") as f:
            f.write(template)

        sv_files = ["ptranspose.sv", "skid.sv"]
        for sv_files in sv_files:
            shutil.copy(f"{rtlsrc}/{sv_files}", code_gen_dir)
        self.set_nodeattr("ipgen_path", code_gen_dir)
        self.set_nodeattr("ip_path", code_gen_dir)

    def get_rtl_file_list(self, abspath=False):
        if abspath:
            code_gen_dir = f"{self.get_nodeattr('code_gen_dir_ipgen')}/"
            rtllib_dir = f'{os.environ["FINN_ROOT"]}/finn-rtllib/ptranspose'
        else:
            code_gen_dir = ''
            rtllib_dir = ''

        return [
            f"{rtllib_dir}/ptranspose.sv",
            f"{rtllib_dir}/skid.sv",
            f"{code_gen_dir}{self.get_nodeattr('gen_top_module')}.v"
        ]

    def code_generation_ipi(self):
        """ Constructs and returns the TCL for node instantiation in Vivado IPI. """
        code_gen_dir = self.get_nodeattr("code_gen_dir_ipgen")
        sourcefiles = [
            "ptranspose.sv",
            "skid.sv",
            f'{self.get_nodeattr("gen_top_module")}.v'
        ]
        sourcefiles = [os.path.join(code_gen_dir, f) for f in sourcefiles]

        cmd = []
        for vf in sourcefiles:
            cmd += [f'add_files -norecurse {vf}']
        cmd += [ f"create_bd_cell -type module -reference {self.get_nodeattr('gen_top_module')} {self.onnx_node.name}" ]
        return cmd

    def execute_node(self, context, graph):
        mode = self.get_nodeattr("exec_mode")
        if mode == "rtlsim" : 
            RTLBackend.execute_node(self, context, graph)
