#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
# @Copyright (c) 2023 Baidu.com, Inc. All Rights Reserved
# @Time    : 2023/5/17 03:00
# @Author  : Liu Tianyuan (liutianyuan02@baidu.com)
# @File    : tensor.py
# @Description    : jax basic operators implementation
"""

"""jax backend basic operators implementation"""

import jax
import jax.numpy as jnp
import numpy as np

lib = jax


# todo: the basic operators are not supported as many as paddle or torch, add more operators

def data_type_dict():
    """
    data_type_dict is a data type object that represents the data type of a tensor.
    """
    return {
        "float16": jnp.float16,
        "float32": jnp.float32,
        "float64": jnp.float64,
        "uint8": jnp.uint8,
        "int8": jnp.int8,
        "int16": jnp.int16,
        "int32": jnp.int32,
        "int64": jnp.int64,
        "bool": jnp.bool_,
        "complex32": jnp.complex32,
        "complex64": jnp.complex64,
        "complex128": jnp.complex128,
    }


def is_tensor(obj):
    """
        is_tensor returns whether obj is a jnp tensor.
    """
    return isinstance(obj, jnp.ndarray)


def shape(input_array):
    """
        shape returns the shape of input_array.
    """
    return input_array.shape


def ndim(input_array):
    """
        ndim returns the dimension of input_array.
        Args:
            input_array: input array
        Returns: dimension of input_array
    """
    return input_array.ndim


def transpose(tensor, axes=None):
    """
        transpose returns a view of the original tensor with its dimensions permuted.
        Args:
            tensor: input tensor
            axes: the permutation of the dimensions of tensor
        Returns: a view of the original tensor with its dimensions permuted
    """
    return jnp.transpose(tensor, axes=axes)


def reshape(tensor, shape):
    """
        reshape returns a tensor with the same data and number of elements as tensor,
        but with the specified shape.
        Args:
            tensor: input tensor
            shape: the new shape
        Returns: a tensor with the same data and number of elements as tensor,
        but with the specified shape
    """
    return jnp.reshape(tensor, shape)


def requires_grad(initial_value, device=None, dtype=None):
    """
        Variable returns a tensor object initialized to initial_value.
        Args:
            initial_value: initial value of the tensor
            dtype: the data type of the returned tensor
        Returns: a tensor object initialized to initial_value
    """
    # Variable returns a tensor object initialized to initial_value.
    return jnp.array(initial_value, dtype=dtype)


def as_tensor(data, device=None, dtype=None):
    """
        as_tensor converts data (numpy array, list, ...) to a jnp tensor.
    """
    # as_tensor converts data (numpy array, list, ...) to a jnp tensor.
    if isinstance(data, jnp.ndarray):
        if dtype is None or data.dtype == dtype:
            return data
        return data.astype(dtype)
    return jnp.asarray(data, dtype=dtype)


def from_numpy(np_array):
    """
        from_numpy converts np_array (numpy array) to a jnp tensor.
    """
    return jnp.asarray(np_array)


def to_numpy(input_tensor):
    """
        to_numpy converts input_tensor (jnp tensor) to a numpy array.
    """
    return np.asarray(input_tensor)


def abs(x):
    """
        abs returns the absolute value of x.
    """
    return jnp.abs(x)


def elu(x):
    """
        elu returns the exponential linear activation of x.
    """
    return jax.nn.elu(x)


def relu(x):
    """
        relu returns the rectified linear unit activation of x.
    """
    return jax.nn.relu(x)


def selu(x):
    """
        selu returns the scaled exponential linear unit activation of x.
    """
    return jax.nn.selu(x)


def sigmoid(x):
    """
        sigmoid returns the sigmoid activation of x.
    """
    return jax.nn.sigmoid(x)


def silu(x):
    """
        silu returns the sigmoid linear unit activation of x.
    """
    return jax.nn.silu(x)


def sin(x):
    """
        sin returns the sine of x.
    """
    return jnp.sin(x)


def square(x):
    """
        square returns the square of x.
    """
    return jnp.square(x)


def tanh(x):
    """
        tanh returns the hyperbolic tangent of x.
    """
    return jnp.tanh(x)


def mean(input_tensor, axis, keepdim=False):
    """
        mean returns the mean value of input_tensor along the axis.
    """
    return jnp.mean(input_tensor, axis=axis, keepdim=keepdim)


def reduce_mean(input_tensor):
    """
        reduce_mean returns the mean value of input_tensor.
    """
    return jnp.mean(input_tensor)


def sum(input_tensor, axis, keepdim=False):
    """
        sum returns the sum of input_tensor along the axis.
    """
    return jnp.sum(input_tensor, axis=axis, keepdim=keepdim)


def reduce_sum(input_tensor):
    """
        reduce_sum returns the sum of input_tensor.
    """
    return jnp.sum(input_tensor)


def zeros(shape, dtype):
    """
        zeros returns a tensor of shape shape filled with zeros.
    """
    return jnp.zeros(shape, dtype=dtype)


def zeros_like(input_tensor):
    """
        zeros_like returns a tensor of shape of input_tensor filled with zeros.
    """
    return jnp.zeros_like(input_tensor)


def ones(shape, dtype):
    """
        ones returns a tensor of shape shape filled with ones.
    """
    return jnp.zeros(shape, dtype=dtype)


def ones_like(input_tensor):
    """
        ones_like returns a tensor of shape of input_tensor filled with ones.
    """
    # ones_like returns a tensor of shape of input_tensor filled with ones.
    return jnp.zeros_like(input_tensor)
