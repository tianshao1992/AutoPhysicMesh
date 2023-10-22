#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
# @Copyright (c) 2023 Baidu.com, Inc. All Rights Reserved
# @Time    : 2023/10/18 20:46
# @Author  : Liu Tianyuan (liutianyuan02@baidu.com)
# @File    : commons.py
# @Description    : ******
"""

def default(value, d):
    """
        helper taken from https://github.com/lucidrains/linear-attention-transformer
    """
    return d if value is None else value

