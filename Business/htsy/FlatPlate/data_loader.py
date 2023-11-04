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

from Dataset.imagefield import ImageFieldDataSet
from Dataset.dataloader import DataLoadersManager


def get_dataloader(config):

    # 原始数据读取
    MeshFile = os.path.join('../data', 'FlatPlate', 'data4', 'Job-article1_new_dnn_better.inp')
    mesh = meshio.read(MeshFile)

    n_s = 1200
    n_x = 9
    n_y = 10
    n_t = 60
    d_t = 10

    # 取出中截面
    middle_set = mesh.points[:, 0] == 0.0
    # 选择中截面索引
    middle_set = np.where(middle_set)[0]
    mesh.point_sets.update({'middle_set': mesh.points[middle_set]})
    # 节点排序
    coord_index = np.lexsort((-mesh.point_sets['middle_set'][:, 2], -mesh.point_sets['middle_set'][:, 1]))
    coords = mesh.point_sets['middle_set'][coord_index].reshape(-1, 1, n_x, n_y, 3)
    # 坐标复制
    coords = np.tile(coords, (n_s, n_t, 1, 1, 1))[..., 1:].astype(np.float32)

    # 读取数据
    f_t = pd.read_csv("../data/FlatPlate/data2/f_t.csv", header=None)
    f_t = np.array(f_t, dtype=np.float32)[:, ::d_t]
    g_t = pd.read_csv("../data/FlatPlate/data2/g_t.csv", header=None)
    g_t = np.array(g_t, dtype=np.float32)[:, ::d_t]
    temper = pd.read_csv("../data/FlatPlate/data2/temper_data.csv", header=None)
    temper = np.array(temper, dtype=np.float32).transpose().reshape(-1, n_x, n_y, n_t, 1).transpose(0, 3, 1, 2, 4)

    # 特征载荷及温度测点
    load_t = np.stack((f_t, g_t), axis=-1)
    inputs = np.concatenate((coords, temper), axis=-1)
    mask = np.zeros_like(inputs[..., 0])
    mask[:, :, config.physics.probe_star::config.physics.probe_step,
               config.physics.probe_star::config.physics.probe_step] = 1.0
    inputs[..., -1] = inputs[..., -1] * mask

    # import matplotlib.pyplot as plt
    # plt.subplot(121)
    # plt.imshow(temper[0, 0, :, :, -1])
    # plt.colorbar()
    # plt.subplot(122)
    # plt.imshow(inputs[0, 0, :, :, -1])
    # plt.colorbar()
    # plt.show()

    # dataloader 制备
    n_train = config.Training.train_size
    n_valid = config.Training.valid_size

    dataset = ImageFieldDataSet(raw_data={'f_t': f_t[:n_train].reshape(-1, 1),
                                          'g_t': g_t[:n_train].reshape(-1, 1),
                                          'load_t': load_t[:n_train].reshape(-1, 2),
                                          'coords': coords[:n_train].reshape(-1, n_x, n_y, 2),
                                          'inputs': inputs[:n_train].reshape(-1, n_x, n_y, 3),
                                          'identify': inputs[:n_train, ..., -1].reshape(-1, n_x*n_y),
                                          'temper': temper[:n_train].reshape(-1, n_x, n_y, 1),
                                          },
                                input_name='inputs',
                                output_name='temper',
    )

    train_loaders = DataLoadersManager(datasets={'reconstruct': dataset(input_name='inputs', output_name='temper'),
                                                 'identify': dataset(input_name='identify', output_name='load_t')},
                                       batch_sizes=config.Training.train_batch_size,
                                       input_transforms={'reconstruct': 'min-max', 'identify': 'min-max'},
                                       output_transforms={'reconstruct': 'min-max', 'identify': 'min-max'},
                                       random_seed=config.Seed, shuffle=True)

    input_transforms = train_loaders.input_transforms
    output_transforms = train_loaders.output_transforms

    dataset = ImageFieldDataSet(raw_data={'f_t': f_t[n_train:n_train+n_valid].reshape(-1, 1),
                                          'g_t': g_t[n_train:n_train+n_valid].reshape(-1, 1),
                                          'load_t': load_t[n_train:n_train+n_valid].reshape(-1, 2),
                                          'coords': coords[n_train:n_train+n_valid].reshape(-1, n_x, n_y, 2),
                                          'inputs': inputs[n_train:n_train+n_valid].reshape(-1, n_x, n_y, 3),
                                          'identify': inputs[n_train:n_train+n_valid, ..., -1].reshape(-1, n_x * n_y),
                                          'temper': temper[n_train:n_train+n_valid].reshape(-1, n_x, n_y, 1),
                                          },
    )

    valid_loaders = DataLoadersManager(datasets={'reconstruct': dataset(input_name='inputs', output_name='temper'),
                                                 'identify': dataset(input_name='identify', output_name='load_t'),},
                                       batch_sizes=config.Training.train_batch_size,
                                       input_transforms=input_transforms,
                                       output_transforms=output_transforms,
                                       random_seed=config.Seed, shuffle=False)

    n_start = n_train+n_valid
    dataset = ImageFieldDataSet(raw_data={'f_t': f_t[n_start:].reshape(-1, 1),
                                          'g_t': g_t[n_start:].reshape(-1, 1),
                                          'load_t': load_t[n_start:].reshape(-1, 2),
                                          'coords': coords[n_start:].reshape(-1, n_x, n_y, 2),
                                          'inputs': inputs[n_start:].reshape(-1, n_x, n_y, 3),
                                          'identify': inputs[n_start:, ..., -1].reshape(-1, n_x * n_y),
                                          'temper': temper[n_start:].reshape(-1, n_x, n_y, 1),
                                          }
    )

    test_loaders = DataLoadersManager(datasets={'reconstruct': dataset(input_name='inputs', output_name='temper'),
                                                'identify': dataset(input_name='identify', output_name='load_t'), },
                                      batch_sizes=n_t,
                                      input_transforms=input_transforms,
                                      output_transforms=output_transforms,
                                      random_seed=config.Seed, shuffle=False)

    return train_loaders, valid_loaders, test_loaders



if __name__ == "__main__":
    from all_config import get_config
    config = get_config()
    train_loaders, valid_loaders, test_loaders = get_dataloader(config)