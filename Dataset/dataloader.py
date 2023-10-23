#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
# @Copyright (c) 2023 Baidu.com, Inc. All Rights Reserved
# @Time    : 2023/10/17 18:07
# @Author  : Liu Tianyuan (liutianyuan02@baidu.com)
# @File    : Samplers.py
# @Description    : ******
"""
import os
import meshio
import gmsh
import numpy as np
from Module import bkd
from Utilizes.commons import default

from torch.utils.data import Dataset, DataLoader


from Dataset.ImageField import ImageFieldDataSet
from Dataset.SpaceMesh import SpaceMeshDataSet

class ImageFieldDataLoader(DataLoader):

    def __init__(self, dataset,
                 *args, **kwargs):
        super(ImageFieldDataLoader, self).__init__(dataset, *args, **kwargs)
        # self.dataset = dataset

        # todo: add sampler


class SpaceMeshDataLoader(DataLoader):

    def __init__(self, dataset,
                 *args, **kwargs):
        super(SpaceMeshDataLoader, self).__init__(dataset, *args, **kwargs)
        # self.dataset = dataset

        # todo: add sampler

if __name__ == '__main__':

    class Mesh(object):
        def __init__(self):
            self.point_sets = {'inlet': np.random.randn(100, 2), 'outlet': np.random.randn(1000, 2)}
            self.field_sets = {'inlet': np.random.randn(100, 3), 'outlet': np.random.randn(1000, 3)}


    mesh = Mesh()
    mesh_dataset = SpaceMeshDataSet(mesh, time_interval=[0, 1])
    outlet_dataset = mesh_dataset('outlet')
    mesh_dataloader = SpaceMeshDataLoader(mesh_dataset('outlet'), batch_size=200)

    for batch in mesh_dataloader:
        print(batch[0].shape)
        break

    mesh_dataset = SpaceMeshDataSet(os.path.join('../demo/OpenAtom/data',
                                                 'raw_mesh',
                                                 'airfoil', 'case1',
                                                 'naca001065_aoa0_box12_4.msh'),
                                    time_interval=[0, 1])
    mesh_dataloader = SpaceMeshDataLoader(mesh_dataset('outlet'), batch_size=200)


    mesh_dataset = SpaceMeshDataSet(os.path.join('../demo/AeroSpaceHeat2D/data',
                                                 'FlatPlate/data4/Job-article1_new_dnn_better.inp'),
                                    time_interval=[0, 1])

    # mesh_dataloader = SpaceMeshDataLoader(mesh_dataset('outlet'), batch_size=200)


    raw_data = {'input': np.random.randn(100, 2), 'output': np.random.randn(100, 3),
                'target': np.random.randn(100, 3)}
    field_dataset = ImageFieldDataSet(raw_data, input_name='input', output_name='output')



