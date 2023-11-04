#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
# @Copyright (c) 2023 Baidu.com, Inc. All Rights Reserved
# @Time    : 2023/10/27 11:39
# @Author  : Liu Tianyuan (liutianyuan02@baidu.com)
# @File    : utils.py
# @Description    : ******
"""

from Module import bkd
import numpy as np


def tensor2numpy(obj):
    """
    convert bkd tensor to numpy.ndarray
    Args:
        obj:
    :return:
        obj: numpy.ndarray
    """
    if bkd.is_tensor(obj):
        obj = obj.cpu().numpy()
    return obj
