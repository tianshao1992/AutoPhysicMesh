#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
# @Copyright (c) 2023 Baidu.com, Inc. All Rights Reserved
# @Time    : 2023/10/22 23:06
# @Author  : Liu Tianyuan (liutianyuan02@baidu.com)
# @File    : SpaceMesh.py
# @Description    : ******
"""

import os
import meshio
import gmsh
import numpy as np
from Module import bkd
from Utilizes.commons import default
from torch.utils.data import Dataset


class SpaceMeshDataSet(Dataset):

    def __init__(self,
                 mesh,
                 train_mode: bool = True,
                 set_name: str = None,
                 time_interval: list or tuple = None,
                 time_slices: int = 1,
                 *args, **kwargs):
        super(SpaceMeshDataSet, self).__init__(*args, **kwargs)

        self._setup(mesh=mesh,
                    train_mode=train_mode,
                    set_name=set_name,
                    time_interval=time_interval,
                    time_slices=time_slices)

    def _setup(self, mesh,
               train_mode: bool,
               set_name: str,
               time_interval: list or tuple,
               time_slices: int):
        # assert isinstance(mesh, (meshio.Mesh)), "mesh should be meshio.Mesh or mesh file"

        if isinstance(mesh, str) and os.path.isfile(mesh):
            mesh = self._loadfile(mesh)

        # else:
        self.mesh = mesh
        if isinstance(mesh, meshio.Mesh):
            self.mesh.point_sets = self._get_point_sets()
            # todo: add field_sets
            self.mesh.field_sets = self.mesh.point_sets
        self.train_mode = train_mode
        self.time_slices = time_slices
        self.time_interval = time_interval

        if set_name is not None:
            assert set_name in self.mesh.point_sets.keys(), "set_name does not exist!"
            self.mesh_points = self.mesh.point_sets[set_name]
            self.mesh_fields = self.mesh.field_sets[set_name]
        else:
            self.mesh_points = None
            self.mesh_fields = None

    def __getitem__(self, idx):
        if self.train_mode:
            if self.time_interval is not None:
                time_stamp = self._get_time_random()
                if isinstance(self.mesh_points, np.ndarray):
                    return (np.concatenate((self.mesh_points[idx], time_stamp), axis=-1),
                            self.mesh_fields[idx],)
                elif isinstance(self.mesh_points, bkd.Tensor):
                    return (bkd.cat((self.mesh_points[idx], bkd.tensor(time_stamp)), dim=-1),
                            self.mesh_fields[idx])
                else:
                    raise NotImplementedError("mesh_points should be np.ndarray or bkd.Tensor")
            else:
                return self.mesh_points[idx], self.mesh_fields[idx]
        else:
            return self.mesh_points[idx], self.mesh_fields[idx]

    def _loadfile(self, file):
        assert os.path.exists(file), 'file not found: ' + file
        try:
            mesh = meshio.read(file)
            print('meshio.read success!')
        except:
            print('meshio.read failed, try gmsh.open and gmsh.write load mesh!')
            gmsh.initialize()
            gmsh.open(file)
            gmsh.write('temp.msh')
            gmsh.finalize()
            mesh = meshio.read('temp.msh')
            os.remove('temp.msh')
        return mesh

    def _get_point_sets(self, ):
        point_sets = {}
        for field_name in self.mesh.field_data.keys():
            point_index = []
            for cell_type, cell_index in self.mesh.cell_sets_dict[field_name].items():
                cell_set = self.mesh.cells_dict[cell_type][cell_index]
                point_index.extend(cell_set.flatten())

            points = self.mesh.points[np.unique(np.array(point_index, dtype=np.int32)), :]
            # # plot fields data
            # if not os.path.exists('mesh_plots'):
            #     os.mkdir('mesh_plots')
            # fig = plt.figure(1)
            # plt.scatter(points[:, 0], points[:, 1], s=0.5)
            # fig.savefig(os.path.join('mesh_plots', field_name + '.jpg'))

            point_sets.update({field_name: points})
        return point_sets

    def _get_time_random(self):
        return np.random.uniform(self.time_interval[0], self.time_interval[1], size=1).astype(np.float32)

    def _get_time_slices(self):
        return np.linspace(self.time_interval[0],
                           self.time_interval[1],
                           num=self.time_slices).astype(np.float32)[..., None] # (N, 1)


    def __call__(self,
                 set_name: str,
                 train_mode: bool = True,
                 time_interval: list or tuple = None,
                 time_slices: int = 1,
                 *args, **kwargs):
        return SpaceMeshDataSet(mesh=self.mesh,
                                set_name=set_name,
                                train_mode=default(train_mode, self.train_mode),
                                time_slices=default(time_slices, self.time_slices),
                                time_interval=default(time_interval, self.time_interval),)

    def __len__(self):
        return len(self.mesh_points)