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


class Mesh:
    def __init__(self, point_sets, field_sets):
        self.point_sets = point_sets
        self.field_sets = field_sets


def get_dataloader():

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

    fields_num = 3
    point_sets = {'all': np.concatenate((coords[None, ...].repeat(t.shape[0], axis=0),
                                         t[..., None, None].repeat(coords.shape[0], axis=1)), axis=-1),
                  'res': coords, 'ics': coords,
                  'inflow': inflow_coords, 'outflow': outflow_coords,
                  'wall': wall_coords, 'cylinder': cylinder_coords}

    field_sets = {'all': np.stack((p_ref, u_ref, v_ref), axis=-1),
                  'res': np.zeros((coords.shape[0], fields_num), dtype=np.float32),
                  'ics': np.stack((p_ref[0], u_ref[0], v_ref[0]), axis=-1),
                  'inflow': np.zeros((inflow_coords.shape[0], fields_num), dtype=np.float32),
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