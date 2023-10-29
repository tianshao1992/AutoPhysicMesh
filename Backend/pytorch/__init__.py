#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
# @Copyright (c) 2023 Baidu.com, Inc. All Rights Reserved
# @Time    : 2023/5/10 02:06
# @Author  : Liu Tianyuan (liutianyuan02@baidu.com)
# @File    : tensor.py
# @Description    : pytorch backend basic operators implementation
"""
# from mpo_engine.backend.pytorch.tensor import *
# from inspect import getmembers, isfunction
# from mpo_engine.backend.pytorch import tensor
#
# functions_list = [o[0] for o in getmembers(tensor) if isfunction(o[1])]
# # print(functions_list)
# __all__ = functions_list

from mpo_engine.backend import _load_mod

# def _load_backend(mod_name):
#     """
#     Load the backend module by name.
#     """
#     # load backend module
#     mod = importlib.import_module(".%s" % mod_name, __name__)
#     # mod = importlib.import_module(mod_name)
#     thismod = sys.modules[__name__]
#     for api, obj in mod.__dict__.items():
#         setattr(thismod, api, obj)


_load_mod(mod_name='tensor', base_name=__name__)


