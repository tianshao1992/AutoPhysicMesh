#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
# @Copyright (c) 2023 Baidu.com, Inc. All Rights Reserved
# @Time    : 2023/10/29 17:00
# @Author  : Liu Tianyuan (liutianyuan02@baidu.com)
# @File    : data_loader.py
# @Description    : ******
"""

from Dataset.spacemesh import SpaceMeshDataSet
from Dataset.dataloader import DataLoadersManager
import numpy as np
from scipy import io as sio

class Mesh:
    def __init__(self, point_sets, field_sets):
        self.point_sets = point_sets
        self.field_sets = field_sets

def get_dataloader(all_config):

    data = sio.loadmat("data/ldc_Re{}.mat".format(all_config.physics.Re))
    u_ref = np.array(data["u"], dtype=np.float32).T
    v_ref = np.array(data["v"], dtype=np.float32).T
    x = np.array(data["x"].flatten(), dtype=np.float32)
    y = np.array(data["y"].flatten(), dtype=np.float32)
    nu = data["nu"]

    grid_x, grid_y = np.meshgrid(x, y)
    coords = np.stack((grid_x, grid_y), axis=-1)
    # top = np.stack([x, np.ones_like(x, dtype=np.float32)], axis=1) # Sample points along the top side (x=1 to x=0, y=1)
    # bottom = np.stack([x, np.zeros_like(x, dtype=np.float32)], axis=1)  # Sample points along the bottom side (x=1 to x=0, y=0)
    # y_ = y
    # y_[-1] = y[-2]
    # left = np.stack([np.zeros_like(y_, dtype=np.float32), y_], axis=1)  # Sample points along the left side (x=0, y=1 to y=0)
    # right = np.stack([np.ones_like(y_, dtype=np.float32), y_], axis=1)  # Sample points along the right side (x=1, y=1 to y=0)

    res_coords = np.stack((grid_x, grid_y), axis=-1)
    bcs_coords = np.stack((coords[:, -1], coords[:, 0], coords[0, :], coords[-1, :]), axis=0)
    res_fields = np.stack((np.zeros_like(u_ref, dtype=np.float32), u_ref, v_ref, ), axis=-1)
    bcs_fields = np.stack((res_fields[:, -1], res_fields[:, 0], res_fields[0, :], res_fields[-1, :]), axis=0) # v = 0 for all walls; u = 1.0 for top and u = 0.0 for the others
    # bcs_fields[0, ..., 1] = 1.0


    point_sets = {'res': res_coords,
                  'bcs': bcs_coords,
                  'all': res_coords[None],
                  }

    field_sets = {'res': res_fields,
                  'bcs': bcs_fields,
                  'all': res_fields[None],
                  }

    all_data = Mesh(point_sets, field_sets)
    dataset = SpaceMeshDataSet(all_data)

    train_datasets = {'bcs': dataset(subset_name='bcs',
                                     train_mode=True),
                      'res': dataset(subset_name='res',
                                     train_mode=True)
                      }

    train_batch_sizes = {'res': all_config.Training.res_batch_size,
                         'bcs': all_config.Training.bcs_batch_size
                         }

    train_loaders = DataLoadersManager(train_datasets,
                                       random_seed=all_config.Seed,
                                       batch_sizes=train_batch_sizes)

    valid_datasets = {'all': dataset(subset_name='all',
                                     train_mode=False),
                      }

    valid_batch_size = all_config.Training.valid_batch_size
    valid_loaders = DataLoadersManager(valid_datasets,
                                       random_seed=all_config.Seed,
                                       batch_sizes=valid_batch_size)

    return train_loaders, valid_loaders


if __name__ == '__main__':
    from demo.ns_ldc_2d.config.default import get_config
    from tabulate import tabulate
    config = get_config()
    train_loaders, valid_loaders = get_dataloader(config)
    train_batch, valid_batch = next(iter(train_loaders)), next(iter(valid_loaders))
    for item in train_batch:
        table = [['input', train_batch[item]['input'].shape],
                 ['target', train_batch[item]['target'].shape]]
        print(item, ':\n ', tabulate(table))

    for item in valid_batch:
        table = [['input', valid_batch[item]['input'].shape],
                 ['target', valid_batch[item]['target'].shape]]
        print(item, ':\n ', tabulate(table))
