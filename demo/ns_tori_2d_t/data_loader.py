#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
# @Copyright (c) 2023 Baidu.com, Inc. All Rights Reserved
# @Time    : 2023/10/29 17:00
# @Author  : Liu Tianyuan (liutianyuan02@baidu.com)
# @File    : data_loader.py
# @Description    : ******
"""

import os
from Dataset.spacemesh import SpaceMeshDataSet
from Dataset.dataloader import DataLoadersManager
import numpy as np

class Mesh:
    def __init__(self, point_sets, field_sets):
        self.point_sets = point_sets
        self.field_sets = field_sets

def get_dataloader(all_config):

    data = np.load("data/ns_tori.npy", allow_pickle=True).item()
    u_ref = np.array(data["u"], dtype=np.float32)
    v_ref = np.array(data["v"], dtype=np.float32)
    w_ref = np.array(data["w"], dtype=np.float32)

    t = np.array(data["t"].flatten(), dtype=np.float32)
    x = np.array(data["x"].flatten(), dtype=np.float32)
    y = np.array(data["y"].flatten(), dtype=np.float32)
    nu = data["viscosity"]

    n_t = t.shape[0]
    n_x = x.shape[0]
    n_y = y.shape[0]

    grid_x, grid_y = np.meshgrid(x, y)

    coords = np.stack((grid_x, grid_y), axis=-1)
    fields = np.stack((u_ref, v_ref, w_ref), axis=-1)[:n_t]


    point_sets = {'res': np.tile(coords, [n_t, 1, 1, 1]),
                  'ics': coords[None],
                  'all': coords,
                  }

    field_sets = {'res': fields,
                  'ics': fields[0:1],
                  'all': fields,
                  }

    all_data = Mesh(point_sets, field_sets)
    dataset = SpaceMeshDataSet(all_data)

    train_datasets = {'ics': dataset(subset_name='ics',
                                     time_interval=[0, 0],
                                     train_mode=True),
                      'res': dataset(subset_name='res',
                                     time_interval=[0, t[n_t-1]],
                                     train_mode=True)
                      }

    train_batch_sizes = {'res': all_config.Training.res_batch_size,
                         'ics': all_config.Training.ics_batch_size
                         }

    train_loaders = DataLoadersManager(train_datasets,
                                       random_seed=all_config.Seed,
                                       batch_sizes=train_batch_sizes)

    valid_datasets = {'all': dataset(subset_name='all',
                                     time_interval=[0, t[n_t-1]],
                                     time_slices=201,
                                     shuffle=True,
                                     train_mode=False),
                      'ics': dataset(subset_name='ics',
                                     time_interval=[0, 0],
                                     time_slices=1,
                                     train_mode=True)
                      }

    valid_batch_size = all_config.Training.valid_batch_size
    valid_loaders = DataLoadersManager(valid_datasets,
                                       random_seed=all_config.Seed,
                                       batch_sizes=valid_batch_size)

    return train_loaders, valid_loaders


if __name__ == '__main__':
    from all_config import get_config
    config = get_config()
    get_dataloader(config)