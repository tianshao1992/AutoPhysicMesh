#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
# @Copyright (c) 2023 Baidu.com, Inc. All Rights Reserved
# @Time    : 2023/5/10 02:06
# @Author  : Liu Tianyuan (liutianyuan02@baidu.com)
# @File    : tensor.py
# @Description    : jax backend basic operators implementation
"""

from mpo_engine.backend import _load_mod

_load_mod('tensor', __name__)