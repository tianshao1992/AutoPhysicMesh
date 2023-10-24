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
from Utilizes.commons import default, identity

from torch.utils.data import Dataset, DataLoader
from torchtnt.utils.data.multi_dataloader import MultiDataLoader

from Dataset.ImageField import ImageFieldDataSet
from Dataset.SpaceMesh import SpaceMeshDataSet

from torchtnt.utils.data.iterators import (
    AllDatasetBatches,
    DataIterationStrategy,
    InOrder,
    MultiIterator,
    RandomizedBatchSampler,
    RoundRobin,
    StoppingMechanism,
)

class DataLoadersManager(MultiDataLoader):

    def __init__(self,
                 datasets,
                 batch_sizes,
                 random_seed: int = 2023,
                 shuffle: bool = True,
                 input_transforms = None,
                 output_transforms = None,
                 samplers=None,
                 *args, **kwargs):
        # super(DataLoadersManager, self).__init__(datasets, *args, **kwargs)
        # self.dataset = dataset

        self.datasets = datasets
        if isinstance(batch_sizes, int):
            batch_size = batch_sizes
            batch_sizes = {}
            for key in datasets.keys():
                batch_sizes[key] = batch_size
        assert batch_sizes.keys() == datasets.keys(), \
            "batch_size keys must be the same as datasets keys!"
        self.batch_sizes = batch_sizes

        if not isinstance(input_transforms, (dict, list, tuple)):
            input_transform = input_transforms
            input_transforms = {}
            for key in datasets.keys():
                input_transforms[key] = default(input_transform, identity)
        assert input_transforms.keys() == datasets.keys(), \
            "input_transforms keys must be the same as datasets keys!"
        self.input_transforms = input_transforms

        if not isinstance(output_transforms, (dict, list, tuple)):
            output_transform = output_transforms
            output_transforms = {}
            for key in datasets.keys():
                output_transforms[key] = default(output_transform, identity)
        assert output_transforms.keys() == datasets.keys(), \
            "output_transforms keys must be the same as datasets keys!"
        self.output_transforms = output_transforms

        dataloaders = {}
        for key, dataset in self.datasets.items():
            # to ensure the same random seed for each dataloader
            g = bkd.Generator()
            g.manual_seed(random_seed)
            dataloaders[key] = \
                DataLoader(dataset, batch_size=self.batch_sizes[key], shuffle=shuffle, generator=g)

            # todo: add samplers
        all_dataset_batches = AllDatasetBatches(
            StoppingMechanism.RESTART_UNTIL_ALL_DATASETS_EXHAUSTED
        )
        super(DataLoadersManager, self).__init__(dataloaders, all_dataset_batches)






if __name__ == '__main__':

    class Mesh(object):
        def __init__(self):
            self.point_sets = {'inlet': np.linspace(0, 100, 100).reshape(-1, 1),
                               'outlet': np.linspace(0, 100, 100).reshape(-1, 1)}
            self.field_sets = {'inlet': np.linspace(0, 100, 100).reshape(-1, 1),
                               'outlet': np.linspace(0, 100, 100).reshape(-1, 1)}


    mesh = Mesh()
    mesh_dataset = SpaceMeshDataSet(mesh, time_interval=None)
    outlet_dataset = mesh_dataset('outlet')

    DataLoaders = DataLoadersManager(datasets={'inlet': mesh_dataset('inlet'), 'outlet': mesh_dataset('outlet')},
                                     batch_sizes={'inlet': 10, 'outlet': 20}, random_seed=2023)
    for epoch in range(2):
        print(epoch)
        for batch in DataLoaders:
            for key, value in batch.items():
                print(key, value[0])


    mesh_dataset = SpaceMeshDataSet(os.path.join('../demo/OpenAtom/data',
                                                 'raw_mesh',
                                                 'airfoil', 'case1',
                                                 'naca001065_aoa0_box12_4.msh'),
                                    time_interval=[0, 1])

    mesh_dataset = SpaceMeshDataSet(os.path.join('../demo/AeroSpaceHeat2D/data',
                                                 'FlatPlate/data4/Job-article1_new_dnn_better.inp'),
                                    time_interval=[0, 1])

    # mesh_dataloader = SpaceMeshDataLoader(mesh_dataset('outlet'), batch_size=200)


    raw_data = {'input': np.random.randn(100, 2), 'output': np.random.randn(100, 3),
                'target': np.random.randn(100, 3)}
    field_dataset = ImageFieldDataSet(raw_data, input_name='input', output_name='output')



