#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
# @Copyright (c) 2023 Baidu.com, Inc. All Rights Reserved
# @Time    : 2023/5/16 01:10
# @Author  : Liu Tianyuan (liutianyuan02@baidu.com)
# @File    : tensor.py
# @Description    : taichi backend basic operators implementation
"""

"""taichi backend basic operators implementation"""
from packaging.version import Version

import taichi as ti

if Version(ti.__version__) < Version("0.4.0"):
    raise RuntimeError("MPO-engine requires PyTorch>=0.4.0")

lib = ti

def data_type_dict():
    """
    data_type_dict is a data type object that represents the data type of a tensor.
    """
    return {
        "float16": ti.float16,
        "float32": ti.float32,
        "float64": ti.float64,
        "uint8": ti.uint8,
        "int8": ti.int8,
        "int16": ti.int16,
        "int32": ti.int32,
        "int64": ti.int64,
        "bool": ti.bool,
        "complex32": ti.complex32,
        "complex64": ti.complex64,
        "complex128": ti.complex128,
    }

# todo: implement all the basic function

