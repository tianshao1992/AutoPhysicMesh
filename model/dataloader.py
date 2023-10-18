#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
# @Copyright (c) 2023 Baidu.com, Inc. All Rights Reserved
# @Time    : 2023/10/17 18:07
# @Author  : Liu Tianyuan (liutianyuan02@baidu.com)
# @File    : Samplers.py
# @Description    : ******
"""

import numpy as np
from model import bkd, nn
from model.commons import default
from torch.utils.data import Dataset, DataLoader, RandomSampler


class SpaceMeshDataSet(Dataset):

    def __init__(self,
                 mesh,
                 train_mode: bool = True,
                 set_name: str = None,
                 time_interval: list or tuple = None,
                 time_slices: int = 1,
                 *args, **kwargs):
        super(SpaceMeshDataSet, self).__init__(*args, **kwargs)

        if set_name is None:
            self.mesh = mesh
            self.time_interval = time_interval
            self.time_slices = time_slices
            self.train_mode = train_mode
            self.mesh_points = None
            self.mesh_fields = None
        else:
            self.mesh = mesh
            self.train_mode = train_mode
            self.time_slices = time_slices
            self.time_interval = time_interval
            self.mesh_points = self.mesh.point_sets[set_name]
            self.mesh_fields = self.mesh.field_sets[set_name]


    def _get_time_random(self):
        return np.random.uniform(self.time_interval[0], self.time_interval[1], size=1).astype(np.float32)

    def _get_time_slices(self):
        return np.linspace(self.time_interval[0],
                           self.time_interval[1],
                           num=self.time_slices).astype(np.float32)[..., None] # (N, 1)

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


class SpaceMeshDataLoader(DataLoader):

    def __init__(self, dataset,
                 *args, **kwargs):
        super(SpaceMeshDataLoader, self).__init__(dataset, *args, **kwargs)
        # self.dataset = dataset

        # todo: add sampler

if __name__ == '__main__':

    class Mesh:
        def __init__(self):
            self.point_sets = {'inlet': np.random.randn(100, 2), 'outlet': np.random.randn(1000, 2)}
            self.field_sets = {'inlet': np.random.randn(100, 3), 'outlet': np.random.randn(1000, 3)}


    mesh = Mesh()
    mesh_dataset = SpaceMeshDataSet(mesh, time_interval=[0, 1])
    outlet_dataset = mesh_dataset('outlet')
    mesh_dataloader = SpaceMeshDataLoader(mesh_dataset('outlet'), batch_size=200)

    for batch in mesh_dataloader:
        print(batch.shape)
        break