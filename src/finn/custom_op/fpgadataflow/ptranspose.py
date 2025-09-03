############################################################################
# Copyright (C) 2025, Advanced Micro Devices, Inc.
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# @author       Shane T. Fleming <shane.fleming@amd.com>
############################################################################
import numpy as np
import warnings
from onnx.helper import make_node
from qonnx.core.datatype import DataType
from finn.custom_op.fpgadataflow.hwcustomop import HWCustomOp


class PTranspose(HWCustomOp):
    """ Abstraction layer for the Parallel 2D transpose. """

    def __init__(self, onnx_node, **kwargs):
        super().__init__(onnx_node, **kwargs)

    def get_nodeattr_types(self):
        my_attrs = {
            "data_type" : ("s", True, ""),
            "in_shape"  : ("ints", True, []), # Needs to be len==2 can we assert that somewhere?
            "SIMD"      : ("i", False, 1)
        }
        my_attrs.update(super().get_nodeattr_types())
        return my_attrs


    def get_normal_input_shape(self, ind=0):
        return self.get_nodeattr("in_shape")

    def get_normal_output_shape(self, ind=0):
        ishape = tuple(self.get_normal_input_shape())
        return ishape[:-2] + (ishape[-1], ishape[-2])

    def get_number_output_values(self): # [STF] Not sure if this is correct
        return int(np.prod(self.get_normal_output_shape())/self.get_nodeattr("SIMD"))

    def execute_node(self, context, graph):
        node = self.onnx_node
        input_data = context[node.input[0]]
        assert len(input_data.shape) >= 2, "PTranspose HWCustomOp requires at least 2D input"
        # Transpose only the last two dimensions: (..., a, b) -> (..., b, a)
        axes = list(range(len(input_data.shape)))
        axes[-2], axes[-1] = axes[-1], axes[-2]
        transposed = np.transpose(input_data, axes)
        context[node.output[0]] = transposed

    def get_input_datatype(self, ind=0):
        data_type = DataType[self.get_nodeattr("data_type")]
        return data_type

    def make_shape_compatible_op(self, model):
        in_shape = self.get_normal_input_shape()
        out_shape = self.get_normal_output_shape()
        return make_node(
            "PTranspose",
            inputs=[self.onnx_node.input[0]],
            outputs=[self.onnx_node.output[0]],
            in_shape=list(in_shape)
        )

    def infer_node_datatype(self, model):
        node = self.onnx_node
        dt = model.get_tensor_datatype(node.input[0])
        if dt != self.get_input_datatype():
            warn_str = f"data_type changing for {node.name}: {str(self.get_input_datatype())} -> {str(dt)}"
            warnings.warn(warn_str)
        self.set_nodeattr("data_type", dt.name)
        model.set_tensor_datatype(node.output[0], dt)

    def verify_node(self):
        raise NotImplementedError("This method is not yet implemented.")

    def get_instream_width(self, ind=0):
        ibits = self.get_input_datatype().bitwidth()
        simd = self.get_nodeattr("SIMD")
        return ibits * simd

    def get_outstream_width(self, ind=0):
        obits = self.get_output_datatype().bitwidth()
        simd = self.get_nodeattr("SIMD")
        return obits * simd

    def get_output_datatype(self, ind=0):
        data_type = DataType[self.get_nodeattr("data_type")]
        return data_type

    def get_folded_output_shape(self, ind=0):
        normal_oshape = list(self.get_normal_output_shape())
        simd = self.get_nodeattr("SIMD")
        assert normal_oshape[-1] % simd == 0, "SIMD must divide into the innermost output dimension"
        fold = int(normal_oshape[-1] / simd)
        folded_oshape = normal_oshape[:-1] + [fold, simd]
        return tuple(folded_oshape)

    def get_folded_input_shape(self, ind=0):
        ## TODO: This one is a litle different to usual because SIMD can span multiple innerdims
        ## For now what I have done is flattened the outer dims
        normal_ishape = list(self.get_normal_input_shape())
        simd = self.get_nodeattr("SIMD")
        fold = int(np.prod(normal_ishape) / simd)
        folded_ishape = [fold , simd]
        return tuple(folded_ishape)

