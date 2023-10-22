#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
# @Copyright (c) 2023 Baidu.com, Inc. All Rights Reserved
# @Time    : 2023/10/16 15:22
# @Author  : Liu Tianyuan (liutianyuan02@baidu.com)
# @File    : activations.py
# @Description    : ******
"""

__all__ = ['get']

from Model import bkd, nn

DICT = {'gelu': nn.GELU(), 'silu': nn.SiLU(), 'relu': nn.ReLU(), 'leakyrelu': nn.LeakyReLU(),
        'elu': nn.ELU(), 'selu': nn.SELU(), 'celu': nn.CELU(), 'hardshrink': nn.Hardshrink(),
        'tanh': nn.Tanh(), 'sigmoid': nn.Sigmoid(), 'softplus': nn.Softplus(), 'softsign': nn.Softsign(),
        'swish': nn.SiLU(), 'mish': nn.Mish(), 'logsigmoid': nn.LogSigmoid(), 'softmin': nn.Softmin(),
        'sin': bkd.sin, 'cos': bkd.cos}

def linear(x):
    # linear activation function
    """
    Linear activation function.
        Args:
            x: Input tensor.
        Returns: Output tensor.
    """
    return x


def get(identifier):
    """Returns function.

    Args:
        identifier: Function or string.

    Returns:
        Function corresponding to the input string or input function.
    """

    if identifier is None:
        return linear
    if isinstance(identifier, str):
        return DICT[identifier]
    if callable(identifier):
        return identifier
    raise TypeError(
        "Could not interpret activation function identifier: {}".format(identifier)
    )


if __name__ == "__main__":

    x = bkd.ones((3, 4), dtype=bkd.float32)
    print(x)
    print(get(None)(x))
    print(get("relu")(x))
    print(get("sigmoid")(x))
    print(get("tanh")(x))
    print(get("swish")(x))
    print(get("silu")(x))
    print(get("gelu")(x))
    print(get("selu")(x))
    print(get("sin")(x))
    print(get("cos")(x))
    print(get("elu")(x))