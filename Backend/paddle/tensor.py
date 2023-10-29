#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
# @Copyright (c) 2023 Baidu.com, Inc. All Rights Reserved
# @Time    : 2023/5/16 01:10
# @Author  : Liu Tianyuan (liutianyuan02@baidu.com)
# @File    : tensor.py
# @Description    : paddle backend basic operators implementation
"""

"""paddle backend basic operators implementation"""
from packaging.version import Version

import paddle

if Version(paddle.__version__) != Version("0.0.0"):
    raise RuntimeError("MPO_engine requires PaddlePaddle==0.0.0(develop).")

if paddle.device.is_compiled_with_cuda():
    paddle.device.set_device("gpu")

lib = paddle


def data_type_dict():
    """
        data_type_dict is a data type object that represents the data type of a tensor.
    """
    return {
        "float16": paddle.float16,
        "float32": paddle.float32,
        "float64": paddle.float64,
        "uint8": paddle.uint8,
        "int8": paddle.int8,
        "int16": paddle.int16,
        "int32": paddle.int32,
        "int64": paddle.int64,
        "bool": paddle.bool,
        # "complex32": paddle.complex32,  # not this is not supported in paddle
        "complex64": paddle.complex64,
        "complex128": paddle.complex128,
    }


def is_gpu_available():
    """
        paddle.device.get_device() returns the current device.
    """
    device = paddle.device.get_device()
    # "cpu"/"gpu:x"/"xpu:x"/"mlu:x"/"npu:x"
    return "gpu" in device


def is_tensor(obj):
    """
        paddle.is_tensor(obj) returns True if obj is a PaddlePaddle tensor.
    """
    return paddle.is_tensor(obj)


def shape(input_tensor):
    """
        paddle.Tensor.shape returns a tuple of tensor dimensions.
    """
    return input_tensor.shape


def size(input_tensor):
    """
        paddle.Tensor.size returns the number of elements in the tensor.
    """
    return int(paddle.numel(input_tensor))


def ndim(input_tensor):
    """
        paddle.Tensor.ndim returns the number of dimensions of a tensor.
    """
    return input_tensor.ndim


def transpose(tensor, axes=None):
    """
        paddle.transpose(tensor, axes) returns a tensor with its dimensions permuted.
    """
    if axes is None:
        axes = tuple(range(tensor.ndim)[::-1])
    return paddle.transpose(tensor, axes)


def reshape(tensor, shape):
    """
        paddle.reshape(tensor, shape) returns a tensor with the same data and number of elements as tensor,
    """
    return paddle.reshape(tensor, shape)


def requires_grad(initial_value, device=None, dtype=None):
    """
        torch.tensor(initial_value, dtype=dtype, requires_grad=True) returns a tensor
        can be calculated gradients with initial_value.
        Args:
            initial_value: A python object, which can be a list, tuple, NumPy ndarray, scalar, and other types.
            dtype: The desired data type of returned tensor.
        Returns: torch.Tensor: A tensor object.
    """
    if paddle.in_dynamic_mode():
        return paddle.to_tensor(initial_value, dtype=dtype, stop_gradient=False)
    return paddle.create_parameter(
        shape=[1],
        dtype=paddle.get_default_dtype() if dtype is None else dtype,
        default_initializer=paddle.nn.initializer.Constant(value=initial_value),
    )


def as_tensor(data, dtype=None, device=None):
    """
        paddle.to_tensor(data, dtype=dtype, device=device) returns a tensor object initialized to data.
    """
    if paddle.is_tensor(data):
        if dtype is None or data.dtype == dtype:
            return data
        return data.astype(dtype)
    return paddle.to_tensor(data, device=device, dtype=dtype)


def sparse_tensor(indices, values, shape):
    """
        paddle.sparse.sparse_coo_tensor(indices, values, shape) returns
        a sparse tensor with non-zero elements at the given indices.
        Args:
            indices: indices of the sparse tensor
            values: values of the sparse tensor
            shape: shape of the sparse tensor
        Returns: a sparse tensor with non-zero elements at the given indices
    """
    # todo: support more format like csr, csc bcsr sparse tensor.
    return paddle.sparse.sparse_coo_tensor(
        list(zip(*indices)), values, shape, stop_gradient=False
    )


def from_numpy(np_array):
    """
        paddle.to_tensor(np_array) returns a tensor from a numpy array.
    """
    return paddle.to_tensor(np_array)


def to_numpy(input_tensor):
    """
        paddle.Tensor.numpy() returns the tensor as a NumPy ndarray.
    """
    return input_tensor.detach().cpu().numpy()


def concat(values, axis):
    """
        paddle.concat(values, axis) returns a tensor obtained by concatenating values along the given axis.
    """
    return paddle.concat(values, axis=axis)


def stack(values, axis):
    """
        paddle.stack(values, axis) returns a tensor obtained by concatenating values along the given axis.
    """
    return paddle.stack(values, axis=axis)


def expand_dims(tensor, axis):
    """
        paddle.expand_dims(tensor, axis) returns a tensor with a length 1 dimension inserted at the specified axis.
    """
    return paddle.unsqueeze(tensor, axis=axis)


def reverse(tensor, axis):
    """
        paddle.reverse(tensor, axis) returns a tensor with the order of elements reversed along the given axis.
    """
    return paddle.flip(tensor, axis)


def roll(tensor, shift, axis):
    """
        paddle.roll(tensor, shift, axis) returns a tensor that rolls the values of a tensor along the given axis.
    """
    return paddle.roll(tensor, shift, axis)


def abs(tensor):
    """
        paddle.abs(tensor) returns a new tensor with the absolute value of the elements of input.
    """
    return paddle.abs(tensor)


def lgamma(tensor):
    """
        paddle.lgamma(tensor) returns a new tensor with the log-gamma function of the elements of input.
    """
    return paddle.lgamma(tensor)


def elu(x):
    """
        paddle.elu(x) returns a new tensor with the exponential linear unit function of the elements of input.
    """
    return paddle.nn.functional.elu(x)


def relu(x):
    """
        paddle.relu(x) returns a new tensor with the rectified linear unit function of the elements of input.
    """
    return paddle.nn.functional.relu(x)


def gelu(x):
    """
        paddle.gelu(x) returns a new tensor with the gaussian error linear unit function of the elements of input.
    """
    return paddle.nn.functional.gelu(x)

def elu(x):
    """
        paddle.elu(x) returns a new tensor with the exponential linear unit function of the elements of input.
    """
    return paddle.nn.functional.elu(x)

def selu(x):
    """
        paddle.selu(x) returns a new tensor with the scaled exponential linear unit function of the elements of input.
    """
    return paddle.nn.functional.selu(x)


def sigmoid(x):
    """
        paddle.sigmoid(x) returns a new tensor with the sigmoid function of the elements of input.
    """
    return paddle.nn.functional.sigmoid(x)


def silu(x):
    """
        paddle.silu(x) returns a new tensor with the sigmoid linear unit function of the elements of input.
    """
    return paddle.nn.functional.silu(x)


def sin(x):
    """
        paddle.sin(x) returns a new tensor with the sine of the elements of input.
    """
    return paddle.sin(x)


def cos(x):
    """
        paddle.cos(x) returns a new tensor with the cosine of the elements of input.
    """
    return paddle.cos(x)


def exp(x):
    """
        paddle.exp(x) returns a new tensor with the exponential of the elements of input.
    """
    return paddle.exp(x)


def square(x):
    """
        paddle.square(x) returns a new tensor with the square of the elements of input.
    """
    return paddle.square(x)


def tanh(x):
    """
        paddle.tanh(x) returns a new tensor with the hyperbolic tangent of the elements of input.
    """
    return paddle.tanh(x)


def pow(x, y):
    """
        paddle.pow(x, y) returns a new tensor with the power of the elements of input.
    """
    return paddle.pow(x, y)


def mean(input_tensor, axis, keepdim=False):
    """
        paddle.mean(input_tensor, axis=axis, keepdim=keepdim) returns the
        mean value of all elements in the input tensor.
    """
    return paddle.mean(input_tensor, axis=axis, keepdim=keepdim)


def reduce_mean(input_tensor):
    """
        paddle.mean(input_tensor) returns the mean value of all elements in the input tensor.
    """
    return paddle.mean(input_tensor)


def sum(input_tensor, axis, keepdim=False):
    """
        paddle.sum(input_tensor, axis=axis, keepdim=keepdim) returns the sum value of all elements in the input tensor.
    """
    return paddle.sum(input_tensor, axis=axis, keepdim=keepdim)


def reduce_sum(input_tensor):
    """
        paddle.sum(input_tensor) returns the sum value of all elements in the input tensor.
    """
    return paddle.sum(input_tensor)


def norm(x, ord=None, axis=None, keepdim=False):
    """
        paddle.norm(x, ord=ord, axis=axis, keepdim=keepdim) returns the norm value of all elements in the input tensor.
    """
    if ord is None:
        ord = 2
    return paddle.linalg.norm(x, p=ord, axis=axis, keepdim=keepdim)


def zeros(shape, dtype):
    """
        paddle.zeros(shape, dtype=dtype) returns a tensor of shape shape filled with 0.
    """
    return paddle.zeros(shape, dtype=dtype)


def zeros_like(input_tensor):
    """
        paddle.zeros_like(input_tensor) returns a tensor of shape shape filled with 0.
    """
    return paddle.zeros_like(input_tensor)


def ones(shape, dtype):
    """
        paddle.ones(shape, dtype=dtype) returns a tensor of shape shape filled with 1.
    """
    return paddle.ones(shape, dtype=dtype)


def ones_like(input_tensor):
    """
        paddle.ones_like(input_tensor) returns a tensor of shape shape filled with 1.
    """
    return paddle.ones_like(input_tensor)


def max(x, axis=None, keepdim=False, return_index=False):
    """
        paddle.max(x, axis=axis, keepdim=keepdim) returns the maximum value of all elements in the input tensor.
    """
    if not return_index:
        return paddle.max(x, axis=axis, keepdim=keepdim)
    return [paddle.max(x, axis=axis, keepdim=keepdim), paddle.argmax(x, axis=axis, keepdim=keepdim)]


def min(x, axis=None, keepdim=False, return_index=False):
    """
        paddle.min(x, axis=axis, keepdim=keepdim) returns the minimum value of all elements in the input tensor.
    """
    if not return_index:
        return paddle.min(x, axis=axis, keepdim=keepdim)[0]
    return [paddle.min(x, axis=axis, keepdim=keepdim), paddle.argmin(x, axis=axis, keepdim=keepdim)]


def maximum(x, y):
    """
        paddle.maximum(x, y) returns a new tensor with the maximum of the elements of input x and y.
    """
    return paddle.maximum(x, y)


def minimum(x, y):
    """
        paddle.minimum(x, y) returns a new tensor with the minimum of the elements of input x and y.
    """
    return paddle.minimum(x, y)


def matmul(x, y):
    """
        paddle.matmul(x, y) returns a new tensor with the matrix multiplication of input x and y.
    """
    return paddle.mm(x, y)


def sparse_dense_matmul(x, y):
    """
        paddle.sparse.matmul(x, y) returns a new tensor with the matrix multiplication of input x and y.
    """
    return paddle.sparse.matmul(x, y)
