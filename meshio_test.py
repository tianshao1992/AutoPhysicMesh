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
import matplotlib.pyplot as plt

from utilizes import get_filelist
from Dataset.data.mesh import MeshLoader


msh_path = os.path.join('data', 'raw_mesh')
msh_files = get_filelist(msh_path)


for file in msh_files:
    msh_model = MeshLoader(file)
    plt.scatter(msh_model.points[:, 0], msh_model.points[:, 1], s=0.1)
    plt.scatter(msh_model.get_point_sets('airfoil')[0][:, 0],
                msh_model.get_point_sets('airfoil')[0][:, 1], s=0.1)
    plt.scatter(msh_model.get_point_sets('inlet')[0][:, 0],
                msh_model.get_point_sets('inlet')[0][:, 1], s=0.1)
    plt.scatter(msh_model.get_point_sets('outlet')[0][:, 0],
                msh_model.get_point_sets('outlet')[0][:, 1], s=0.1)
    plt.show()






