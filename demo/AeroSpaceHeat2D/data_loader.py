#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
# @Copyright (c) 2023 Baidu.com, Inc. All Rights Reserved
# @Time    : 2023/10/18 11:08
# @Author  : Liu Tianyuan (liutianyuan02@baidu.com)
# @File    : data_loader.py
# @Description    : ******
"""

import os
import pandas as pd
import numpy as np
import meshio

from Dataset.dataprocess import DataNormer
from Dataset.dataloader import ImageFieldDataSet, ImageFieldDataLoader

def get_dataloader():

    MeshFile = os.path.join('data', 'FlatPlate', 'data4', 'Job-article1_new_dnn_better.inp')
    mesh = meshio.read(MeshFile)

    # 取出中截面
    middle_set = mesh.points[:, 0] == 0.0
    # 选择中截面索引
    middle_set = np.where(middle_set)[0]
    mesh.point_sets.update({'middle_set': mesh.points[middle_set]})
    # 节点排序
    coord_index = np.lexsort((-mesh.point_sets['middle_set'][:, 2], -mesh.point_sets['middle_set'][:, 1]))
    coords = mesh.point_sets['middle_set'][coord_index].reshape(-1, 1, 9, 10, 3)
    # 坐标复制
    coords = np.tile(coords, (1200, 60, 1, 1, 1))[..., 1:].astype(np.float32)

    # 读取数据
    f_t = pd.read_csv("data/FlatPlate/data2/f_t.csv", header=None)
    f_t = np.array(f_t, dtype=np.float32)[:, ::10]
    g_t = pd.read_csv("data/FlatPlate/data2/g_t.csv", header=None)
    g_t = np.array(g_t, dtype=np.float32)[:, ::10]
    temper = pd.read_csv("data/FlatPlate/data2/temper_data.csv", header=None)
    temper = np.array(temper, dtype=np.float32).transpose().reshape(-1, 9, 10, 60, 1).transpose(0, 3, 1, 2, 4)

    load_t = np.stack((f_t, g_t), axis=-1)
    inputs = np.concatenate((coords, np.tile(load_t[:, :, None, None, :], (9, 10, 1))), axis=-1)

    # import matplotlib.pyplot as plt
    # plt.imshow(temper[0, :, :, 0])
    # plt.colorbar()
    # plt.show()

    n_train = 1000
    n_valid = 100

    dataset = ImageFieldDataSet(raw_data={'f_t': f_t[:n_train], 'g_t': g_t[:n_train],
                                          'load_t': np.stack((f_t, g_t), axis=-1)[:n_train],
                                          'coords': coords[:n_train],
                                          'inputs': inputs[:n_train],
                                          'temper': temper[:n_train]},
                                input_name='inputs',
                                output_name='temper',
    )



    train_loaders = {'temper': ImageFieldDataLoader(dataset(input_name='inputs', output_name='temper'),
                                                    batch_size=4,
                                                    shuffle=True,)

    }

    dataset = ImageFieldDataSet(raw_data={'f_t': f_t[n_train:], 'g_t': g_t[n_train:],
                                          'load_t': np.stack((f_t, g_t), axis=-1)[n_train:],
                                          'coords': coords[n_train:],
                                          'inputs': inputs[n_train:],
                                          'temper': temper[n_train:]},
                                input_name='inputs',
                                output_name='temper',
    )

    valid_loaders = {'temper': ImageFieldDataLoader(dataset(input_name='inputs', output_name='temper'),
                                                    batch_size=1,
                                                    shuffle=False,)

    }

    return train_loaders, valid_loaders



if __name__ == "__main__":

    data_loaders = get_dataloader()