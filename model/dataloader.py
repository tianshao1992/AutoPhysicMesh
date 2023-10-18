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
from torch.utils.data import Dataset, DataLoader, RandomSampler


class SpaceMeshDataSet(Dataset):

    def __init__(self, mesh, *args, **kwargs):
        super(SpaceMeshDataSet, self).__init__(*args, **kwargs)
        self.mesh = mesh
        self.time_interval = None
        self.mesh_points = self.mesh.point_sets
        self.mesh_fields = self.mesh.field_sets

    def _get_time(self):
        return np.random.uniform(self.time_interval[0], self.time_interval[1], size=1)

    def __getitem__(self, idx):
        if self.time_interval is not None:
            time_stamp = self._get_time()
            if isinstance(self.mesh_points, np.ndarray):
                return np.concatenate((self.mesh_points[idx], time_stamp), axis=-1), self.mesh_fields[idx],
            elif isinstance(self.mesh_points, bkd.Tensor):
                return bkd.cat((self.mesh_points[idx], bkd.tensor(time_stamp)), dim=-1), self.mesh_fields[idx]
        else:
            return self.mesh_points[idx], self.mesh_fields[idx]

    def __len__(self):
        return len(self.mesh_points)


class SpaceMeshDataLoader(DataLoader):

    def __init__(self, dataset, set_name, time_interval, *args, **kwargs):
        super(SpaceMeshDataLoader, self).__init__(dataset, *args, **kwargs)
        # self.dataset = dataset
        self.dataset.mesh_points = self.dataset.mesh.point_sets[set_name]
        self.dataset.mesh_fields = self.dataset.mesh.field_sets[set_name]
        self.dataset.time_interval = time_interval
        # todo: add sampler

if __name__ == '__main__':

    class Mesh:
        def __init__(self):
            self.point_sets = {'inlet': np.random.randn(100, 2), 'outlet': np.random.randn(1000, 2)}
            self.field_sets = {'inlet': np.random.randn(100, 3), 'outlet': np.random.randn(1000, 3)}


    mesh = Mesh()
    mesh_dataset = SpaceMeshDataSet(mesh)
    mesh_dataloader = SpaceMeshDataLoader(mesh_dataset, 'outlet', batch_size=200)

    for batch in mesh_dataloader:
        # print(batch.shape)
        break


    mesh = Mesh()
    mesh_dataset = SpaceMeshDataSet(mesh, time_interval=[0, 1])
    mesh_dataloader = SpaceMeshDataLoader(mesh_dataset, 'outlet', batch_size=200)

    for batch in mesh_dataloader:
        print(batch.shape)
        break