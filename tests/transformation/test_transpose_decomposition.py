############################################################################
# Copyright (C) 2025, Advanced Micro Devices, Inc.
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# @author       Shane T. Fleming <shane.fleming@amd.com>
############################################################################

import pytest

import numpy as np
import os
import tempfile
import torch
from brevitas.export import export_qonnx
from qonnx.core.datatype import DataType
from qonnx.core.modelwrapper import ModelWrapper
from qonnx.transformation.infer_datatypes import InferDataTypes
from qonnx.transformation.infer_shapes import InferShapes
from qonnx.util.cleanup import cleanup as qonnx_cleanup
from torch import nn
from typing import Optional, Tuple

import finn.core.onnx_exec as oxe
from finn.transformation.fpgadataflow.transpose_decomposition import (
    TransposeDecomposition,
)


class PytorchShuffle(nn.Module):
    """From pytorch create a reshape and transpose combination for testing."""

    def __init__(
        self,
        transpose_perm: Tuple[int, ...],
        reshape1_shape: Optional[Tuple[int, ...]] = None,
        reshape2_shape: Optional[Tuple[int, ...]] = None,
    ) -> None:
        super().__init__()
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


def export_py_to_onnx_file(
    model: nn.Module,
    input_shape: Tuple[int, ...],
    onnx_path: str,
    opset_version: int = 17,
):
    """
    Export the given PyTorch module to ONNX file using brevitas.export_qonnx
    to keep parity with your original pipeline.
    """
    model_input = torch.randn(*input_shape)
    export_qonnx(model, model_input, onnx_path, opset_version=opset_version)
    qonnx_cleanup(onnx_path)


def build_qonnx_model_from_file(onnx_path: str, dt: DataType) -> ModelWrapper:
    """
    Load ONNX file into qonnx ModelWrapper and infer shapes/types.
    """
    mw = ModelWrapper(onnx_path)
    mw.set_tensor_datatype(mw.graph.input[0].name, dt)
    mw.set_tensor_datatype(mw.graph.output[0].name, dt)
    mw = mw.transform(InferShapes())
    mw = mw.transform(InferDataTypes())
    return mw


def construct_onnx_models(
    input_shape: Tuple[int, ...],
    transpose_perm: Tuple[int, ...],
    reshape1_shape: Optional[Tuple[int, ...]],
    reshape2_shape: Optional[Tuple[int, ...]],
    dt: DataType,
) -> Tuple[str, str]:
    """
    Returns (onnx_before_path, onnx_after_path).
    - onnx_before_path: file path to exported ONNX (original).
    - onnx_after_path: file path to ONNX after applying TransposeDecomposition via qonnx.
    Temporary files are created and returned (caller/test will remove them).
    """
    torch_model = PytorchShuffle(
        transpose_perm=transpose_perm,
        reshape1_shape=reshape1_shape,
        reshape2_shape=reshape2_shape,
    )

    fd_before, path_before = tempfile.mkstemp(suffix=".onnx")
    os.close(fd_before)
    fd_after, path_after = tempfile.mkstemp(suffix=".onnx")
    os.close(fd_after)

    try:
        export_py_to_onnx_file(torch_model, input_shape, path_before)

        mw_before = build_qonnx_model_from_file(path_before, dt)

        mw_after = mw_before.transform(TransposeDecomposition())

        mw_before.save(path_before)
        mw_after.save(path_after)

        return path_before, path_after
    except Exception:
        if os.path.exists(path_before):
            os.remove(path_before)
        if os.path.exists(path_after):
            os.remove(path_after)
        raise


@pytest.mark.transform
@pytest.mark.parametrize(
    "perm",
    [
        [0, 1, 2, 3],
        [1, 0, 2, 3],
        [0, 1, 3, 2],
        [1, 0, 3, 2],
    ],
)
@pytest.mark.parametrize(
    "input_shape",
    [
        (1, 2, 3, 4),
        (2, 8, 16, 4),
        (1, 32, 32, 3),
    ],
)
def test_transpose_decomposition_4d(perm, input_shape):
    dt = DataType["FLOAT32"]

    torch_model = PytorchShuffle(transpose_perm=tuple(perm))

    fd, path_before = tempfile.mkstemp(suffix=".onnx")
    os.close(fd)

    try:
        export_py_to_onnx_file(torch_model, input_shape, path_before)

        model_before = build_qonnx_model_from_file(path_before, dt)
        model_before = model_before.transform(InferShapes())

        model_after = model_before.transform(TransposeDecomposition())
        model_after = model_after.transform(InferShapes())

        np.random.seed(0)
        input_array = np.random.randn(*input_shape).astype(np.float32)
        input_dict = {model_before.graph.input[0].name: input_array}

        assert oxe.compare_execution(model_before, model_after, input_dict)
    finally:
        if os.path.exists(path_before):
            os.remove(path_before)


@pytest.mark.transform
@pytest.mark.parametrize(
    "perm",
    [
        [0, 1, 2],
        [0, 2, 1],
    ],
)
@pytest.mark.parametrize(
    "input_shape",
    [
        (4, 8, 16),
        (1, 32, 64),
        (2, 4, 8),
    ],
)
def test_transpose_decomposition_3d(perm, input_shape):
    dt = DataType["FLOAT32"]

    torch_model = PytorchShuffle(transpose_perm=tuple(perm))

    fd, path_before = tempfile.mkstemp(suffix=".onnx")
    os.close(fd)

    try:
        export_py_to_onnx_file(torch_model, input_shape, path_before)

        model_before = build_qonnx_model_from_file(path_before, dt)
        model_before = model_before.transform(InferShapes())

        model_after = model_before.transform(TransposeDecomposition())
        model_after = model_after.transform(InferShapes())

        np.random.seed(0)
        input_array = np.random.randn(*input_shape).astype(np.float32)
        input_dict = {model_before.graph.input[0].name: input_array}

        assert oxe.compare_execution(model_before, model_after, input_dict)
    finally:
        if os.path.exists(path_before):
            os.remove(path_before)


@pytest.mark.transform
@pytest.mark.parametrize(
    "perm",
    [
        [0, 1, 2, 3, 4],
        [4, 0, 1, 2, 3],
        [1, 3, 0, 4, 2],
        [2, 4, 1, 3, 0],
    ],
)
def test_transpose_decomposition_5d(perm):
    input_shape = (2, 3, 4, 5, 6)
    dt = DataType["FLOAT32"]

    torch_model = PytorchShuffle(transpose_perm=tuple(perm))

    fd, path_before = tempfile.mkstemp(suffix=".onnx")
    os.close(fd)

    try:
        export_py_to_onnx_file(torch_model, input_shape, path_before)

        model_before = build_qonnx_model_from_file(path_before, dt)
        model_before = model_before.transform(InferShapes())

        model_after = model_before.transform(TransposeDecomposition())
        model_after = model_after.transform(InferShapes())

        np.random.seed(0)
        input_array = np.random.randn(*input_shape).astype(np.float32)
        input_dict = {model_before.graph.input[0].name: input_array}

        assert oxe.compare_execution(model_before, model_after, input_dict)
    finally:
        if os.path.exists(path_before):
            os.remove(path_before)


@pytest.mark.transform
@pytest.mark.parametrize(
    "perm",
    [
        [1, 0],
        [0, 1],
    ],
)
def test_transpose_decomposition_2d(perm):
    input_shape = (8, 16)
    dt = DataType["FLOAT32"]

    torch_model = PytorchShuffle(transpose_perm=tuple(perm))

    fd, path_before = tempfile.mkstemp(suffix=".onnx")
    os.close(fd)

    try:
        export_py_to_onnx_file(torch_model, input_shape, path_before)

        model_before = build_qonnx_model_from_file(path_before, dt)
        model_before = model_before.transform(InferShapes())

        model_after = model_before.transform(TransposeDecomposition())
        model_after = model_after.transform(InferShapes())

        np.random.seed(0)
        input_array = np.random.randn(*input_shape).astype(np.float32)
        input_dict = {model_before.graph.input[0].name: input_array}

        assert oxe.compare_execution(model_before, model_after, input_dict)
    finally:
        if os.path.exists(path_before):
            os.remove(path_before)
