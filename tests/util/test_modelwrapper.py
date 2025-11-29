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

import onnx
import onnx.helper as oh
from qonnx.core.modelwrapper import ModelWrapper

import finn.core  # noqa: F401


@pytest.mark.util
def test_get_global_in_out():
    # Create a simple ONNX model for testing
    inp = oh.make_tensor_value_info("test_input", onnx.TensorProto.FLOAT, [1, 4])
    outp = oh.make_tensor_value_info("test_output", onnx.TensorProto.FLOAT, [1, 4])

    identity_node = oh.make_node("Identity", ["test_input"], ["test_output"])

    graph = oh.make_graph([identity_node], "test_graph", [inp], [outp])
    onnx_model = oh.make_model(graph, producer_name="finn-test")
    model = ModelWrapper(onnx_model)

    # Test get_global_in
    assert model.get_global_in() == "test_input"

    # Test get_global_out
    assert model.get_global_out() == "test_output"

    # Verify these match the old pattern
    assert model.get_global_in() == model.graph.input[0].name
    assert model.get_global_out() == model.graph.output[0].name
