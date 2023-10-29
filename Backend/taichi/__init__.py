#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
# @Copyright (c) 2023 Baidu.com, Inc. All Rights Reserved
# @Time    : 2023/9/5 12:21
# @Author  : Liu Tianyuan (liutianyuan02@baidu.com)
# @File    : __init__.py.py
# @Description    : ******
"""

from mpo_engine.backend import _load_mod

_load_mod('tensor', __name__)