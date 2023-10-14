#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
# @Copyright (c) 2023 Baidu.com, Inc. All Rights Reserved
# @Time    : 2023/10/13 15:22
# @Author  : Liu Tianyuan (liutianyuan02@baidu.com)
# @File    : meshio_test.py
# @Description    : ******
"""


import os

import gmsh
import meshio
import numpy as np


def get_filelist(dir):
    Filelist = []
    for home, dirs, files in os.walk(dir):
        for filename in files:
            # 文件名列表，包含完整路径
            if filename.endswith('.msh'):
                Filelist.append(os.path.join(home, filename))
            # # 文件名列表，只包含文件名
            # Filelist.append( filename)
    return Filelist

msh_path = os.path.join('data', 'raw_mesh')

msh_files = get_filelist(msh_path)
for file in msh_files:
    mesh = meshio.read(file)

