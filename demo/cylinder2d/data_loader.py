#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
# @Copyright (c) 2023 Baidu.com, Inc. All Rights Reserved
# @Time    : 2023/10/18 11:08
# @Author  : Liu Tianyuan (liutianyuan02@baidu.com)
# @File    : data_loader.py
# @Description    : ******
"""


from Dataset.SpaceMesh import SpaceMeshDataSet
from Dataset.dataloader import DataLoadersManager
import numpy as np
import meshio

class Mesh:
    def __init__(self, point_sets, field_sets):
        self.point_sets = point_sets
        self.field_sets = field_sets

def inflow_fn(y, fields_num):
    U_max = 0.3
    u = 4 * U_max * y * (0.41 - y) / (0.41 ** 2)
    puv = np.zeros(y.shape[:-1] + (fields_num,), dtype=np.float32)
    puv[..., (1,)] = u
    return puv

def Nondimension(L_star, U_star, x, type='x'):
    if type == 'x':
        x = x / L_star
    elif type == 'y':
        x[:, 0] = x[:, 0] / U_star ** 2
        x[:, 1:] = x[:, 1:] / U_star
    return x


def get_dataloader(all_config):

    non_dim = lambda x, type='x': Nondimension(all_config.physics.L_star, all_config.physics.U_star, x, type)
    fields_num = 3
    data = np.load("data/ns_steady.npy", allow_pickle=True).item()
    u_ref = np.array(data["u"], dtype=np.float32)
    v_ref = np.array(data["v"], dtype=np.float32)
    p_ref = np.array(data["p"], dtype=np.float32)
    coords = np.array(data["coords"], dtype=np.float32)

    inflow_coords = np.array(data["inflow_coords"], dtype=np.float32)
    outflow_coords = np.array(data["outflow_coords"], dtype=np.float32)
    wall_coords = np.array(data["wall_coords"], dtype=np.float32)
    cylinder_coords = np.array(data["cylinder_coords"], dtype=np.float32)

    nu = np.array(data["nu"], dtype=np.float32)

    point_sets = {'all': non_dim(coords),
                  'res': non_dim(coords),
                  'inflow': non_dim(inflow_coords),
                  'outflow': non_dim(outflow_coords),
                  'wall': non_dim(wall_coords),
                  'cylinder': non_dim(cylinder_coords)
                  }


    all_config.physics.L = point_sets['res'][..., 0].max().item() - point_sets['res'][..., 0].min().item()
    all_config.physics.W = point_sets['res'][..., 1].max().item() - point_sets['res'][..., 1].min().item()

    field_sets = {'all': non_dim(np.stack((p_ref, u_ref, v_ref), axis=-1), type='y'),
                  'res': non_dim(np.stack((p_ref, u_ref, v_ref), axis=-1), type='y'),
                  'inflow': non_dim(inflow_fn(inflow_coords[..., (1,)], fields_num), type='y'),
                  'outflow': non_dim(np.zeros((outflow_coords.shape[0], fields_num), dtype=np.float32), type='y'),
                  'wall': non_dim(np.zeros((wall_coords.shape[0], fields_num), dtype=np.float32), type='y'),
                  'cylinder': non_dim(np.zeros((cylinder_coords.shape[0], fields_num), dtype=np.float32), type='y')
                  }

    all_data = Mesh(point_sets, field_sets)
    dataset = SpaceMeshDataSet(all_data)

    train_datasets = {'res': dataset(set_name='res'),
                      'inflow': dataset(set_name='inflow'),
                      'outflow': dataset(set_name='outflow'),
                      'wall': dataset(set_name='wall'),
                      'cylinder': dataset(set_name='cylinder')
    }

    train_batch_sizes = {'res': all_config.training.res_batch_size,
                         'inflow': all_config.training.inflow_batch_size,
                         'outflow': all_config.training.outflow_batch_size,
                         'wall': all_config.training.wall_batch_size,
                         'cylinder': all_config.training.cylinder_batch_size}

    train_loaders = DataLoadersManager(train_datasets, random_seed=2023, batch_sizes=train_batch_sizes)

    valid_datasets = {'all': dataset(set_name='all', train_mode=False,)}

    valid_loaders = DataLoadersManager(valid_datasets, random_seed=2023, batch_sizes=len(field_sets['all']))

    return train_loaders, valid_loaders




if __name__ == "__main__":
    from all_config import get_config

    all_config = get_config()
    data_loaders = get_dataloader(all_config)
    a = 1