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

class Mesh:
    def __init__(self, point_sets, field_sets):
        self.point_sets = point_sets
        self.field_sets = field_sets

def inflow_fn(y, fields_num):
    U_max = 1.5
    u = 4 * U_max * y * (0.41 - y) / (0.41 ** 2)
    puv = np.zeros(y.shape[:-1] + (fields_num,), dtype=np.float32)
    puv[..., (1,)] = u
    return puv

def Nondimension(L_star, U_star, x, type='x'):
    if type == 'x':
        x[..., :1] = x[..., :1] / L_star  # T_star = 1.0
    elif type == 'y':
        x[..., 0] = x[..., 0] / U_star ** 2
        x[..., 1:] = x[..., 1:] / U_star
    return x


def get_dataloader(all_config):

    non_dim = lambda x, type='x': Nondimension(all_config.physics.L_star, all_config.physics.U_star, x, type)
    fields_num = 3
    data = np.load("data/ns_unsteady.npy", allow_pickle=True).item()
    u_ref = np.array(data["u"], dtype=np.float32)
    v_ref = np.array(data["v"], dtype=np.float32)
    p_ref = np.array(data["p"], dtype=np.float32)
    t = np.array(data["t"], dtype=np.float32)
    coords = np.array(data["coords"], dtype=np.float32)
    inflow_coords = np.array(data["inflow_coords"], dtype=np.float32)
    outflow_coords = np.array(data["outflow_coords"], dtype=np.float32)
    wall_coords = np.array(data["wall_coords"], dtype=np.float32)
    cylinder_coords = np.array(data["cylinder_coords"], dtype=np.float32)

    nu = np.array(data["nu"], dtype=np.float32)


    point_sets = {'all': np.concatenate((coords[None, ...].repeat(t.shape[0], axis=0),
                                         t[..., None, None].repeat(coords.shape[0], axis=1)), axis=-1),
                  'res': coords, 'ics': coords,
                  'inflow': inflow_coords, 'outflow': outflow_coords,
                  'wall': wall_coords, 'cylinder': cylinder_coords}
    point_sets.update({key: non_dim(value) for key, value in point_sets.items()})

    all_config.physics.L = point_sets['res'][..., 0].max().item() - point_sets['res'][..., 0].min().item()
    all_config.physics.W = point_sets['res'][..., 1].max().item() - point_sets['res'][..., 1].min().item()
    all_config.physics.T = t.max().item()

    field_sets = {'all': np.stack((p_ref, u_ref, v_ref), axis=-1),
                  'res': np.zeros((coords.shape[0], fields_num), dtype=np.float32),
                  # Use the last time step of a coarse numerical solution as the initial condition
                  'ics': np.stack((p_ref[-1], u_ref[-1], v_ref[-1]), axis=-1),
                  'inflow': inflow_fn(inflow_coords[..., (1,)], fields_num),
                  'outflow': np.zeros((outflow_coords.shape[0], fields_num), dtype=np.float32),
                  'wall': np.zeros((wall_coords.shape[0], fields_num), dtype=np.float32),
                  'cylinder': np.zeros((cylinder_coords.shape[0], fields_num), dtype=np.float32)}
    field_sets.update({key: non_dim(value, type='y') for key, value in field_sets.items()})

    all_data = Mesh(point_sets, field_sets)
    dataset = SpaceMeshDataSet(all_data)

    train_datasets = {'res': dataset(set_name='res', time_interval=(0, t[-1])),
                     'ics': dataset(set_name='ics', time_interval=(0, 0)),
                     'inflow': dataset(set_name='inflow', time_interval=(0, t[-1])),
                     'outflow': dataset(set_name='outflow', time_interval=(0, t[-1])),
                     'wall': dataset(set_name='wall', time_interval=(0, t[-1])),
                     'cylinder': dataset(set_name='cylinder', time_interval=(0, t[-1]))
    }

    train_batch_sizes = {'res': all_config.training.res_batch_size,
                        'ics': all_config.training.ic_batch_size,
                        'inflow': all_config.training.inflow_batch_size,
                        'outflow': all_config.training.outflow_batch_size,
                        'wall': all_config.training.wall_batch_size,
                        'cylinder': all_config.training.cylinder_batch_size}


    train_loaders = DataLoadersManager(train_datasets, random_seed=2023, batch_sizes=train_batch_sizes)

    valid_datasets = {'all': dataset(set_name='all', time_interval=(0, t[-1]), train_mode=False,)}

    valid_loaders = DataLoadersManager(valid_datasets, random_seed=2023, batch_sizes=1)

    return train_loaders, valid_loaders


def get_fine_mesh():
    data = np.load("data/fine_mesh.npy", allow_pickle=True).item()
    fine_coords = np.array(data["coords"], dtype=np.float32)

    data = np.load("data/fine_mesh_near_cylinder.npy", allow_pickle=True).item()
    fine_coords_near_cyl = np.array(data["coords"], dtype=np.float32)

    return fine_coords, fine_coords_near_cyl



if __name__ == "__main__":
    from all_config import get_config
    all_config = get_config()
    train_loaders, valid_loaders = get_dataloader(all_config)

    for batch in train_loaders:
        print(batch['res']['input'].shape)
        print(batch['res']['target'].shape)
        print(batch['ics']['input'].shape)
        print(batch['ics']['target'].shape)
        print(batch['inflow']['input'].shape)
        print(batch['inflow']['target'].shape)
        print(batch['outflow']['input'].shape)
        print(batch['outflow']['target'].shape)
        print(batch['wall']['input'].shape)
        print(batch['wall']['target'].shape)
        print(batch['cylinder']['input'].shape)
        print(batch['cylinder']['target'].shape)
        break

    fine_coords, fine_coords_near_cyl = get_fine_mesh()