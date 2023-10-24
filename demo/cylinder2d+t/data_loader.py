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
    U_max = 1.5
    u = 4 * U_max * y * (0.41 - y) / (0.41 ** 2)
    puv = np.zeros(y.shape[:-1] + (fields_num,), dtype=np.float32)
    puv[..., (1,)] = u
    return puv


def get_dataloader(all_config):


    data = np.load("data/ns_unsteady.npy", allow_pickle=True).item()
    u_ref = np.array(data["u"], dtype=np.float32) / all_config.physics.U_star
    v_ref = np.array(data["v"], dtype=np.float32) / all_config.physics.U_star
    p_ref = np.array(data["p"], dtype=np.float32) / all_config.physics.U_star**2
    t = np.array(data["t"], dtype=np.float32)
    coords = np.array(data["coords"], dtype=np.float32) / all_config.physics.L_star
    inflow_coords = np.array(data["inflow_coords"], dtype=np.float32) / all_config.physics.L_star
    outflow_coords = np.array(data["outflow_coords"], dtype=np.float32) / all_config.physics.L_star
    wall_coords = np.array(data["wall_coords"], dtype=np.float32) / all_config.physics.L_star
    cylinder_coords = np.array(data["cylinder_coords"], dtype=np.float32) / all_config.physics.L_star
    nu = np.array(data["nu"], dtype=np.float32)
    all_config.physics.L = coords[..., 0].max().item()
    all_config.physics.W = coords[..., 1].max().item()
    all_config.physics.T = t.max().item()

    fields_num = 3
    point_sets = {'all': np.concatenate((coords[None, ...].repeat(t.shape[0], axis=0),
                                         t[..., None, None].repeat(coords.shape[0], axis=1)), axis=-1),
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
    train_loaders = {'res': SpaceMeshDataLoader(dataset(set_name='res', time_interval=(0, t[-1])),
                                                batch_size=4096, shuffle=True),
                   'ics': SpaceMeshDataLoader(dataset(set_name='ics', time_interval=(0, 0)),
                                              batch_size=1024, shuffle=True),
                   'inflow': SpaceMeshDataLoader(dataset(set_name='inflow', time_interval=(0, t[-1])),
                                                 batch_size=1024, shuffle=True),
                   'outflow': SpaceMeshDataLoader(dataset(set_name='outflow', time_interval=(0, t[-1])),
                                                  batch_size=1024, shuffle=True),
                   'wall': SpaceMeshDataLoader(dataset(set_name='wall', time_interval=(0, t[-1])),
                                               batch_size=1024, shuffle=True),
                   'cylinder': SpaceMeshDataLoader(dataset(set_name='cylinder', time_interval=(0, t[-1])),
                                                   batch_size=1024, shuffle=True),

    }

    valid_loaders = {'all': SpaceMeshDataLoader(dataset(set_name='all', time_interval=(0, t[-1]), train_mode=False,),
                                               batch_size=1,
                                               shuffle=False)}

    return train_loaders, valid_loaders


def get_fine_mesh():
    data = np.load("data/fine_mesh.npy", allow_pickle=True).item()
    fine_coords = np.array(data["coords"], dtype=np.float32)

    data = np.load("data/fine_mesh_near_cylinder.npy", allow_pickle=True).item()
    fine_coords_near_cyl = np.array(data["coords"], dtype=np.float32)

    return fine_coords, fine_coords_near_cyl



if __name__ == "__main__":

    data_loaders = get_dataloader()
    fine_coords, fine_coords_near_cyl = get_fine_mesh()