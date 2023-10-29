#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
# @Copyright (c) 2023 Baidu.com, Inc. All Rights Reserved
# @Time    : 2023/5/10 02:06
# @Author  : Liu Tianyuan (liutianyuan02@baidu.com)
# @File    : tensor.py
# @Description    : pytorch backend basic operators implementation
"""

"""pytorch backend basic operators implementation"""
from packaging.version import Version
import torch


if Version(torch.__version__) < Version("1.10.0"):
    raise RuntimeError("MPO-engine requires PyTorch>=1.10.0.")

# To write device-agnostic (CPU or GPU) code, a common pattern is to first determine
# torch.device and then use it for all the tensors.
# https://pytorch.org/docs/stable/notes/cuda.html
# >>> device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# >>> tensor.to(device=device)
# But, taking care of all tensors requires a lot of work.

if torch.cuda.is_available():
    torch.set_default_tensor_type(torch.cuda.FloatTensor)

lib = torch

def data_type_dict():
    """
    data_type_dict is a data type object that represents the data type of a tensor.
    """
    return {
        "float16": torch.float16,
        "float32": torch.float32,
        "float64": torch.float64,
        "uint8": torch.uint8,
        "int8": torch.int8,
        "int16": torch.int16,
        "int32": torch.int32,
        "int64": torch.int64,
        "bool": torch.bool,
        "complex32": torch.complex32,
        "complex64": torch.complex64,
        "complex128": torch.complex128,
    }


def is_gpu_available():

    """
        torch.cuda.is_available() returns True if there are available GPUs.
    """
    return torch.cuda.is_available()


def is_tensor(obj):
    """
        torch.is_tensor(obj) returns True if obj is a PyTorch tensor.
        Args:
            obj: A python object to check.
        Returns: bool: True if obj is a PyTorch tensor.
    """
    return torch.is_tensor(obj)


def shape(input_tensor):
    """
        torch.Tensor.shape returns a tuple of tensor dimensions.
        Args:
            input_tensor: A tensor object.
        Returns: tuple: A tuple of integers describing the shape of input_tensor.
    """
    return list(input_tensor.shape)


def size(tensor):
    """
        torch.numel(tensor) returns the number of elements in a tensor.
        Args:
            tensor: A tensor object.
        Returns: int: The number of elements in tensor.
    """
    return torch.numel(tensor)


def ndim(input_tensor):
    """
        torch.Tensor.dim returns the number of tensor dimensions.
        Args:
            input_tensor: A tensor object.
        Returns: int: The number of tensor dimensions.
    """
    return input_tensor.dim()


def transpose(tensor, axes=None):
    """
        torch.permute(tensor, axes) returns a view of the original tensor with its dimensions permuted.
        Args:
            tensor: A tensor object.
            axes: A permutation of the dimensions of tensor.
        Returns: torch.Tensor: A tensor object.
    """
    if axes is None:
        axes = tuple(range(tensor.dim())[::-1])
    return torch.permute(tensor, axes)


def reshape(tensor, shape):
    """
        torch.reshape(tensor, shape) returns a tensor with the same data and number of elements as tensor,
        but with the specified shape. When possible, the returned tensor will be a view of input.
        Otherwise, it will be a copy. Contiguous inputs and inputs with compatible strides can be reshaped
        without copying, but you should not depend on the copying vs. viewing behavior.
        Args:
            tensor: A tensor object.
            shape: A list of integers, the shape of the output tensor.
        Returns: torch.Tensor: A tensor object.
    """
    return torch.reshape(tensor, shape)


def requires_grad(initial_value, device=None, dtype=None):
    """
        torch.tensor(initial_value, dtype=dtype, requires_grad=True) returns a tensor
        can be calculated gradients with initial_value.
        Args:
            initial_value: A python object, which can be a list, tuple, NumPy ndarray, scalar, and other types.
            device: The desired device of returned tensor.
            dtype: The desired data type of returned tensor.
        Returns: torch.Tensor: A tensor object.
    """

    return torch.tensor(initial_value, device=device, dtype=dtype, requires_grad=True)


def as_tensor(data, dtype=None, device=None):
    """
        torch.as_tensor(data, dtype=dtype, device=device) returns a tensor with the same data and dtype as data.
        Args:
            data: A python object, which can be a list, tuple, NumPy ndarray, scalar, and other types.
            dtype: The desired data type of returned tensor.
            device: The desired device of returned tensor.
        Returns: torch.Tensor: A tensor object.
    """
    if isinstance(data, torch.Tensor):
        if dtype is None or data.dtype == dtype:
            return data
        return data.type(dtype=dtype)
    return torch.as_tensor(data, dtype=dtype, device=device)


def sparse_tensor(indices, values, shape):
    """
        torch.sparse_coo_tensor(indices, values, shape, requires_grad=True) returns a sparse tensor.
        Args:
            indices: A tuple of Tensors, each one containing indices for a specific dimension.
            values: A Tensor containing values for each set of indices.
            shape: A list of integers, the shape of the sparse tensor.
        Returns: torch.Tensor: A tensor object.
    """
    # todo: support more format like csr, csc bcsr sparse tensor.
    return torch.sparse_coo_tensor(list(zip(*indices)), values, shape, requires_grad=True)


def from_numpy(np_array):
    """
        torch.from_numpy(np_array) returns a tensor from a numpy.ndarray.
        note: Both torch.from_numpy and torch.as_tensor work without memory copy.
        https://discuss.pytorch.org/t/from-numpy-vs-as-tensor/79932
        https://stackoverflow.com/questions/48482787/pytorch-memory-model-torch-from-numpy-vs-torch-tensor
        But torch.from_numpy cannot handle device.
        Args:
            np_array: A numpy.ndarray object.
        Returns: torch.Tensor: A tensor object.
    """
    return torch.as_tensor(np_array)


def to_numpy(input_tensor):
    """
        torch.Tensor.numpy() returns a numpy.ndarray object with the same data as the tensor.
        Args:
            input_tensor: A tensor object.
        Returns: numpy.ndarray: A numpy.ndarray object.
    """
    return input_tensor.detach().cpu().numpy()


def concat(values, axis):
    """
        torch.cat(values, axis) concatenates the given sequence of seq tensors in the given dimension.
        Args:
            values: A sequence of seq tensors.
            axis: An integer, the axis to concatenate along.
        Returns: torch.Tensor: A tensor object.
    """
    return torch.cat(values, dim=axis)


def stack(values, axis):
    """
        torch.stack(values, axis) concatenates sequence of tensors along a new dimension.
        Args:
            values: A sequence of seq tensors.
            axis: An integer, the axis to stack along.
        Returns: torch.Tensor: A tensor object.
    """
    return torch.stack(values, dim=axis)


def expand_dims(tensor, axis):
    """
        torch.unsqueeze(tensor, axis) returns a new tensor with a dimension
        of size one inserted at the specified position.
        Args:
            tensor: A tensor object.
            axis: An integer, the axis to expand along.
        Returns: torch.Tensor: A tensor object.
    """
    # torch.unsqueeze(tensor, axis) returns a new tensor with a dimension of size one inserted at the specified position.
    return torch.unsqueeze(tensor, dim=axis)


def reverse(tensor, axis):
    """
        torch.flip(tensor, axis) returns a tensor that is reversed with respect to axis.
        Args:
            tensor: A tensor object.
            axis: An integer, the axis to reverse along.
        Returns: torch.Tensor: A tensor object.
    """
    return torch.flip(tensor, dims=axis)


def roll(tensor, shift, axis):
    """
        torch.roll(tensor, shift, axis) returns a tensor where each dim is shifted by shift along the given axis.
        Args:
            tensor: A tensor object.
            shift: An integer, the number of places by which the elements of the tensor are shifted.
            axis: An integer, the axis to roll along.
        Returns: torch.Tensor: A tensor object.
    """
    return torch.roll(tensor, shift, dims=axis)


def abs(x):
    """
        torch.abs(x) returns a new tensor with the absolute values of the elements of input.
        Args:
            x: A tensor object.
        Returns: torch.Tensor: A tensor object.
    """
    return torch.abs(x)

def lgamma(x):
    """
        torch.lgamma(x) returns the log gamma function of input.
        Args:
            x: A tensor object.
        Returns: torch.Tensor: A tensor object.
    """
    # torch.lgamma(x) returns the log gamma function of input.
    return torch.lgamma(x)


def elu(x):
    """
        torch.nn.functional.elu(x) returns a new tensor with the exponential linear function of the elements of input.
        Args:
            x: A tensor object.
        Returns: torch.Tensor: A tensor object.
    """
    # torch.nn.functional.elu(x) returns a new tensor with the exponential linear function of the elements of input.
    return torch.elu(x)


def relu(x):
    """
        torch.nn.functional.relu(x) returns a new tensor with the
        rectified linear unit function of the elements of input.
        Args:
            x: A tensor object.
        Returns: torch.Tensor: A tensor object.
    """
    return torch.relu(x)


def selu(x):
    """
        torch.nn.functional.selu(x) returns a new tensor with the scaled exponential linear unit function
        of the elements of input.
        Args:
            x: A tensor object.
        Returns: torch.Tensor: A tensor object.
    """
    # torch.nn.functional.selu(x) returns a new tensor with the scaled exponential linear unit function
    # of the elements of input.
    return torch.selu(x)


def sigmoid(x):
    """
        torch.nn.functional.sigmoid(x) returns a new tensor with the sigmoid function of the elements of input.
        Args:
            x: A tensor object.
        Returns: torch.Tensor: A tensor object.
    """
    # torch.nn.functional.sigmoid(x) returns a new tensor with the sigmoid function of the elements of input.
    return torch.sigmoid(x)


def silu(x):
    """
        torch.nn.functional.silu(x) returns a new tensor with the sigmoid linear unit function of the elements of input.
        Args:
            x: A tensor object.
        Returns: torch.Tensor: A tensor object.
    """
    # torch.nn.functional.silu(x) returns a new tensor with the sigmoid linear unit function of the elements of input.
    return torch.nn.functional.silu(x)


def sin(x):
    """
        torch.sin(x) returns a new tensor with the sine of the elements of input.
        Args:
            x: A tensor object.
        Returns: torch.Tensor: A tensor object.
    """
    # torch.sin(x) returns a new tensor with the sine of the elements of input.
    return torch.sin(x)


def cos(x):
    """
        torch.cos(x) returns a new tensor with the cosine of the elements of input.
        Args:
            x: A tensor object.
        Returns: torch.Tensor: A tensor object.
    """
    # torch.cos(x) returns a new tensor with the cosine of the elements of input.
    return torch.cos(x)


def exp(x):
    """
        torch.exp(x) returns a new tensor with the exponential of the elements of input.
        Args:
            x: A tensor object.
        Returns: torch.Tensor: A tensor object.
    """
    # torch.exp(x) returns a new tensor with the exponential of the elements of input.
    return torch.exp(x)


def square(x):
    """
        torch.square(x) returns a new tensor with the square of the elements of input.
        Args:
            x: A tensor object.
        Returns: torch.Tensor: A tensor object.
    """
    # torch.square(x) returns a new tensor with the square of the elements of input.
    return torch.square(x)


def tanh(x):
    """
        torch.tanh(x) returns a new tensor with the hyperbolic tangent of the elements of input.
        Args:
            x: A tensor object.
        Returns: torch.Tensor: A tensor object.
    """
    return torch.tanh(x)

def gelu(x):
    """
        torch.nn.functional.gelu(x) returns a new tensor with the Gaussian Error Linear Units
        function of the elements of input.
        Args:
            x: A tensor object.
        Returns: torch.Tensor: A tensor object.
    """
    return torch.nn.functional.gelu(x)


def elu(x):
    """
        torch.nn.functional.elu(x) returns a new tensor with the exponential linear function of the elements of input.
        Args:
            x: A tensor object.
        Returns: torch.Tensor: A tensor object.
    """
    return torch.nn.functional.elu(x)


def pow(x, y):
    """
        torch.pow(x, y) returns a new tensor with the elements of input x to the power of y.
        Args:
            x: A tensor object.
            y: A tensor object or float.
        Returns: torch.Tensor: A tensor object.
    """
    # torch.pow(x, y) returns a new tensor with the elements of input x to the power of y.
    return torch.pow(x, y)


def mean(input_tensor, axis, keepdim=False):
    """
        torch.mean(input_tensor, axis, keepdim=keepdim) returns the mean value of all elements in the input tensor.
        Args:
            input_tensor: A tensor object.
            axis: An integer, the axis to reduce along.
            keepdim: If this is set to True, the axes which are reduced are left
            in the result as dimensions with size one.
        Returns: torch.Tensor: A tensor object.
    """
    return torch.mean(input_tensor, dim=axis, keepdim=keepdim)


def reduce_mean(input_tensor):
    """
        torch.mean(input_tensor) returns the mean value of all elements in the input tensor.
        Args:
            input_tensor: A tensor object.
        Returns: torch.Tensor: A tensor object.
    """
    # torch.mean(input_tensor) returns the mean value of all elements in the input tensor.
    return torch.mean(input_tensor)


def sum(input_tensor, axis, keepdim=False):
    """
        torch.sum(input_tensor, axis, keepdim=keepdim) returns the sum value of all elements in the input tensor.
        Args:
            input_tensor: A tensor object.
            axis: An integer, the axis to reduce along.
            keepdim: If this is set to True, the axes which are reduced are left
            in the result as dimensions with size one.
        Returns: torch.Tensor: A tensor object.
    """
    # torch.sum(input_tensor, axis, keepdim=keepdim) returns the sum value of all elements in the input tensor.
    return torch.sum(input_tensor, dim=axis, keepdim=keepdim)


def reduce_sum(input_tensor):
    """
        torch.sum(input_tensor) returns the sum value of all elements in the input tensor.
        Args:
            input_tensor: A tensor object.
        Returns: torch.Tensor: A tensor object.
    """
    return torch.sum(input_tensor)


def norm(tensor, ord=None, axis=None, keepdim=False):
    """
        torch.linalg.norm(tensor, ord=ord, dim=axis, keepdim=keepdim) returns the matrix norm or vector norm of a tensor.
        # norm value of all elements in the input tensor.
        # ord: The order of norm. Supported values are 'fro', 'nuc', 1, 2, np.inf and any positive
        # real number yielding the corresponding p-norm. Default is None, in which case the Frobenius norm is computed.
        # axis: If axis is an integer, it specifies the axis of x along which to compute the vector norms.
        # If axis is a 2-tuple, it specifies the axes that hold 2-D matrices,
        # and the matrix norms of these matrices are computed.
    """

    return torch.linalg.norm(tensor, ord=ord, dim=axis, keepdim=keepdim)


def zeros(shape, dtype):
    """
        torch.zeros(shape, dtype=dtype) returns a tensor filled with the scalar value 0,
        with the shape defined by the variable argument size.
        Args:
            shape: A list of integers, the shape of the output tensor.
            dtype: The desired data type of returned tensor.
        Returns: torch.Tensor: A tensor object.
    """
    # torch.zeros(shape, dtype=dtype) returns a tensor filled with the scalar value 0,
    # with the shape defined by the variable argument size.
    return torch.zeros(shape, dtype=dtype)


def zeros_like(input_tensor):
    """
        torch.zeros_like(input_tensor) returns a tensor filled with the scalar value 0,
        with the same size as input_tensor.
        Args:
            input_tensor: A tensor object.
        Returns: torch.Tensor: A tensor object.
    """
    return torch.zeros_like(input_tensor)


def ones(shape, dtype):
    """
        torch.ones(shape, dtype=dtype) returns a tensor filled with the scalar value 1,
        with the shape defined by the variable argument size.
        Args:
            shape: A list of integers, the shape of the output tensor.
            dtype: The desired data type of returned tensor.
        Returns: torch.Tensor: A tensor object.
    """
    return torch.ones(shape, dtype=dtype)


def ones_like(input_tensor):
    """
        torch.ones_like(input_tensor) returns a tensor filled with the scalar value 1,
        with the same size as input_tensor.
        Args:
            input_tensor: A tensor object.
        Returns: torch.Tensor: A tensor object.
    """
    # torch.ones_like(input_tensor) returns a tensor filled with the scalar value 1,
    return torch.ones_like(input_tensor)

def max(x, axis=None, keepdim=False, return_index=False):
    """
        torch.max(x, dim=axis, keepdim=keepdim) returns the maximum value of all elements in the input tensor.
        Args:
            x: A tensor object.
            axis: An integer, the axis to reduce along.
            keepdim: If this is set to True, the axes which are reduced
            are left in the result as dimensions with size one.
            return_index: If this is set to True, the indices of the elements in the original input tensor are returned.
        Returns: torch.Tensor: A tensor object.
    """
    # torch.max(x, dim=axis, keepdim=keepdim) returns the maximum value of all elements in the input tensor.
    if not return_index:
        return torch.max(x, dim=axis, keepdim=keepdim)[0]
    return torch.max(x, dim=axis, keepdim=keepdim)

def min(x, axis=None, keepdim=False, return_index=False):
    """
        torch.min(x, dim=axis, keepdim=keepdim) returns the minimum value of all elements in the input tensor.
        Args:
            x: A tensor object.
            axis: An integer, the axis to reduce along.
            keepdim: If this is set to True, the axes which are reduced are left in the result as dimensions with size one.
            return_index: If this is set to True, the indices of the elements in the original input tensor are returned.
        Returns: torch.Tensor: A tensor object.
    """
    if not return_index:
        return torch.min(x, dim=axis, keepdim=keepdim)[0]
    return torch.min(x, dim=axis, keepdim=keepdim)


def maximum(x, y):
    """
        torch.maximum(x, y) returns the maximum value of two tensors.
        Args:
            x: A tensor object.
            y: A tensor object.
        Returns: torch.Tensor: A tensor object.
    """
    return torch.maximum(x, y)

def minimum(x, y):
    """
        torch.minimum(x, y) returns the minimum value of two tensors.
        Args:
            x: A tensor object.
            y: A tensor object.
        Returns: torch.Tensor: A tensor object.
    """
    # torch.minimum(x, y) returns the minimum value of two tensors.
    return torch.minimum(x, y)

def matmul(x, y):
    """
        torch.matmul(x, y) returns a tensor product of two tensors with broadcast mechanism.
        Args:
            x: A tensor object.
            y: A tensor object.
        Returns: torch.Tensor: A tensor object.
    """
    return torch.mm(x, y)


def sparse_dense_matmul(x, y):
    """
        torch.sparse.mm(x, y) returns a sparse tensor product of two tensors with broadcast mechanism.
        Args:
            x: A sparse tensor object.
            y: A tensor object.
        Returns: torch.Tensor: A tensor object.
    """
    # torch.sparse.mm(x, y) returns a sparse tensor product of two tensors with broadcast mechanism.
    return torch.sparse.mm(x, y)


