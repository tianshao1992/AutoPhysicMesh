#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
# @Copyright (c) 2023 Baidu.com, Inc. All Rights Reserved
# @Time    : 2023/10/26 10:34
# @Author  : Liu Tianyuan (liutianyuan02@baidu.com)
# @File    : odesolvers.py
# @Description    : ******
"""

import scipy
import torchdiffeq # todo: support more backend

__all__ = ['get']

from Module import bkd, nn

DICT = {'torch_integrator': torchdiffeq.odeint, 'scipy_integrator': scipy.integrate.odeint}


def get(identifier):
    """Returns function.

    Args:
        identifier: Function or string.

    Returns:
        Function corresponding to the input string or input function.
    """

    if identifier is None:
        raise ValueError
    if isinstance(identifier, str):
        return DICT[identifier]
    if callable(identifier):
        return identifier
    raise TypeError(
        "Could not interpret the ode solvers: {}".format(identifier)
    )
