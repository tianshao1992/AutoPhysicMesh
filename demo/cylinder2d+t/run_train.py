#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
# @Copyright (c) 2023 Baidu.com, Inc. All Rights Reserved
# @Time    : 2023/10/17 0:13
# @Author  : Liu Tianyuan (liutianyuan02@baidu.com)
# @File    : run_train.py
# @Description    : ******
"""

from model import bkd, nn
from network import NavierStokes2D
from model.dataloader import SpaceMeshDataSet, SpaceMeshDataLoader
import numpy as np
import meshio


class Mesh:
    def __init__(self, point_sets, field_sets):
        self.point_sets = point_sets
        self.field_sets = field_sets


def parabolic_inflow(y, U_max):
    u = 4 * U_max * y * (0.41 - y) / (0.41**2)
    v = np.zeros_like(y)
    return u, v

def get_dataset():

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
    point_sets = {'res': coords, 'ics': coords,
                  'inflow': inflow_coords, 'outflow': outflow_coords,
                  'wall': wall_coords, 'cylinder': cylinder_coords}

    field_sets = {'res': np.zeros((coords.shape[0], fields_num), dtype=np.float32),
                  'ics': np.stack((p_ref[0], u_ref[0], v_ref[0]), axis=-1),
                  'inflow': np.zeros((inflow_coords.shape[0], fields_num), dtype=np.float32),
                  'outflow': np.zeros((outflow_coords.shape[0], fields_num), dtype=np.float32),
                  'wall': np.zeros((wall_coords.shape[0], fields_num), dtype=np.float32),
                  'cylinder': np.zeros((cylinder_coords.shape[0], fields_num), dtype=np.float32)}

    all_data = Mesh(point_sets, field_sets)

    dataset = SpaceMeshDataSet(all_data)
    data_loader = {'res': SpaceMeshDataLoader(dataset, set_name='res', time_interval=(0, t[-1]),
                                              batch_size=1000, shuffle=True),
                   'ics': SpaceMeshDataLoader(dataset, set_name='ics', time_interval=(0, 0),
                                              batch_size=1000, shuffle=True),
                   'inflow': SpaceMeshDataLoader(dataset, set_name='inflow', time_interval=(0, t[-1]),
                                              batch_size=1000, shuffle=True),
                   'outflow': SpaceMeshDataLoader(dataset, set_name='outflow', time_interval=(0, t[-1]),
                                              batch_size=1000, shuffle=True),
                   'wall': SpaceMeshDataLoader(dataset, set_name='wall', time_interval=(0, t[-1]),
                                              batch_size=1000, shuffle=True),
                   'cylinder': SpaceMeshDataLoader(dataset, set_name='cylinder', time_interval=(0, t[-1]),
                                              batch_size=1000, shuffle=True),

    }

    return data_loader


def get_fine_mesh():
    data = np.load("data/fine_mesh.npy", allow_pickle=True).item()
    fine_coords = np.array(data["coords"], dtype=np.float32)

    data = np.load("data/fine_mesh_near_cylinder.npy", allow_pickle=True).item()
    fine_coords_near_cyl = np.array(data["coords"], dtype=np.float32)

    return fine_coords, fine_coords_near_cyl


res = get_dataset()
fine_coords, fine_coords_near_cyl = get_fine_mesh()


import matplotlib.pyplot as plt
plt.scatter(res[3][:, 0], res[3][:, 1], s=0.1)
plt.scatter(fine_coords[:, 0], fine_coords[:, 1], s=0.1)
plt.scatter(fine_coords_near_cyl[:, 0], fine_coords_near_cyl[:, 1], s=0.1)
plt.axis('equal')
plt.show()