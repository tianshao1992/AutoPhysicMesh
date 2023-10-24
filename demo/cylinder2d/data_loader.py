#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
# @Copyright (c) 2023 Baidu.com, Inc. All Rights Reserved
# @Time    : 2023/10/18 11:08
# @Author  : Liu Tianyuan (liutianyuan02@baidu.com)
# @File    : data_loader.py
# @Description    : ******
"""


from Dataset.dataloader import SpaceMeshDataSet, SpaceMeshDataLoader
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


def get_dataloader(all_config):


    data = np.load("data/ns_steady.npy", allow_pickle=True).item()
    u_ref = np.array(data["u"], dtype=np.float32) / all_config.physics.U_star
    v_ref = np.array(data["v"], dtype=np.float32) / all_config.physics.U_star
    p_ref = np.array(data["p"], dtype=np.float32) / all_config.physics.U_star**2
    coords = np.array(data["coords"], dtype=np.float32) / all_config.physics.L_star
    inflow_coords = np.array(data["inflow_coords"], dtype=np.float32) / all_config.physics.L_star
    outflow_coords = np.array(data["outflow_coords"], dtype=np.float32) / all_config.physics.L_star
    wall_coords = np.array(data["wall_coords"], dtype=np.float32) / all_config.physics.L_star
    cylinder_coords = np.array(data["cylinder_coords"], dtype=np.float32) / all_config.physics.L_star
    nu = np.array(data["nu"], dtype=np.float32)
    all_config.physics.L = coords[..., 0].max().item()
    all_config.physics.W = coords[..., 1].max().item()

    fields_num = 3
    point_sets = {'all': coords[None, ...],
                  'res': coords, 'ics': coords,
                  'inflow': inflow_coords, 'outflow': outflow_coords,
                  'wall': wall_coords, 'cylinder': cylinder_coords}

    field_sets = {'all': np.stack((p_ref, u_ref, v_ref), axis=-1),
                  'res': np.zeros((coords.shape[0], fields_num), dtype=np.float32),
                  # Use the last time step of a coarse numerical solution as the initial condition
                  'ics': np.stack((p_ref[-1], u_ref[-1], v_ref[-1]), axis=-1),
                  'inflow': inflow_fn(inflow_coords[..., (1,)] * all_config.physics.L_star, fields_num),
                  'outflow': np.zeros((outflow_coords.shape[0], fields_num), dtype=np.float32),
                  'wall': np.zeros((wall_coords.shape[0], fields_num), dtype=np.float32),
                  'cylinder': np.zeros((cylinder_coords.shape[0], fields_num), dtype=np.float32)}

    all_data = Mesh(point_sets, field_sets)

    dataset = SpaceMeshDataSet(all_data)
    train_loaders = {'res': SpaceMeshDataLoader(dataset(set_name='res', ),
                                                batch_size=4096, shuffle=True),
                   'ics': SpaceMeshDataLoader(dataset(set_name='ics', ),
                                              batch_size=1024, shuffle=True),
                   'inflow': SpaceMeshDataLoader(dataset(set_name='inflow', ),
                                                 batch_size=1024, shuffle=True),
                   'outflow': SpaceMeshDataLoader(dataset(set_name='outflow', ),
                                                  batch_size=1024, shuffle=True),
                   'wall': SpaceMeshDataLoader(dataset(set_name='wall', ),
                                               batch_size=1024, shuffle=True),
                   'cylinder': SpaceMeshDataLoader(dataset(set_name='cylinder', ),
                                                   batch_size=1024, shuffle=True),

    }

    valid_loaders = {'all': SpaceMeshDataLoader(dataset(set_name='all', train_mode=False,),
                                               batch_size=1,
                                               shuffle=False)}

    return train_loaders, valid_loaders




if __name__ == "__main__":
    from all_config import get_config

    all_config = get_config()
    data_loaders = get_dataloader(all_config)
    a = 1