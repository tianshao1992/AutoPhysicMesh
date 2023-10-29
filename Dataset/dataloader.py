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
import numpy as np
from Module import bkd
from Utilizes.commons import default, identity

# todo: use bkd.DataLoader in the future
from torch.utils.data import Dataset, DataLoader
from torchtnt.utils.data.multi_dataloader import MultiDataLoader

from Dataset.imagefield import ImageFieldDataSet
from Dataset.spacemesh import SpaceMeshDataSet
from Dataset.timeseries import TimeSeriesDataSet
from Dataset.preprocess import DataNormer


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
                 input_transforms=None,
                 output_transforms=None,
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

        self.data_transforms_collect(input_transforms, output_transforms)

        dataloaders = {}
        for key, dataset in self.datasets.items():
            # to ensure the same random seed for each dataloader
            g = bkd.Generator()
            g.manual_seed(random_seed)
            dataloaders[key] = \
                DataLoader(dataset, batch_size=self.batch_sizes[key], shuffle=shuffle, generator=g)

            # todo: add samplers
        if isinstance(datasets, TimeSeriesDataSet):
            all_dataset_batches = RoundRobin(
                StoppingMechanism.ALL_DATASETS_EXHAUSTED
            )
        else:
            all_dataset_batches = AllDatasetBatches(
                StoppingMechanism.RESTART_UNTIL_ALL_DATASETS_EXHAUSTED
            )
        super(DataLoadersManager, self).__init__(dataloaders, all_dataset_batches)

    def data_transforms_collect(self, input_transforms, output_transforms):

        # todo: add a series of input_transforms and output_transforms
        if not isinstance(input_transforms, (dict, list, tuple)):
            input_transform = input_transforms
            input_transforms = {}
            for key in self.datasets.keys():
                input_transforms[key] = default(input_transform, DataNormer(data=None, method=None))
        assert input_transforms.keys() == self.datasets.keys(), \
            "input_transforms keys must be the same as datasets keys!"
        for key, value in input_transforms.items():
            if isinstance(value, str):
                input_transforms[key] = DataNormer(self.datasets[key].input_data, method=value)
        self.input_transforms = input_transforms

        if not isinstance(output_transforms, (dict, list, tuple)):
            output_transform = output_transforms
            output_transforms = {}
            for key in self.datasets.keys():
                output_transforms[key] = default(output_transform, DataNormer(data=None, method=None))
        assert output_transforms.keys() == self.datasets.keys(), \
            "output_transforms keys must be the same as datasets keys!"
        for key, value in output_transforms.items():
            if isinstance(value, str):
                output_transforms[key] = DataNormer(self.datasets[key].output_data, method=value)
        self.output_transforms = output_transforms

    def batch_preprocess(self, batch):
        batch = self.feature_transforms_run(batch, mode='norm', label_name='input', transform='input')
        batch = self.feature_transforms_run(batch, mode='norm', label_name='target', transform='output')
        return batch

    def batch_postprocess(self, batch):
        # batch = self.input_transforms_run(batch, mode='back', label_name='input')
        # batch = self.output_transforms_run(batch, mode='back', label_name='target')
        try:
            batch = self.feature_transforms_run(batch, mode='back', label_name='pred', transform='output')
        except:
            batch = self.feature_transforms_run(batch, mode='back', label_name='pred_norm', transform='output')
        return batch

    def feature_transforms_run(self, batch, mode, label_name, transform):

        for key, value in batch.items():
            if label_name in value.keys():
                if mode == 'norm':
                    postfix = '_norm'
                    if transform == 'input':
                        norm_value = self.input_transforms[key].norm(value[label_name])
                    elif transform == 'output':
                        norm_value = self.output_transforms[key].norm(value[label_name])
                elif mode == 'back':
                    postfix = ''
                    if transform == 'input':
                        norm_value = self.input_transforms[key].back(value[label_name])
                    elif transform == 'output':
                        norm_value = self.output_transforms[key].back(value[label_name])
                else:
                    raise ValueError("the mode is not supported!")
            else:
                norm_value = None

            batch[key].update({label_name + postfix: norm_value})

        return batch




if __name__ == '__main__':

    from Dataset.imagefield import ImageFieldDataSet
    from Dataset.spacemesh import SpaceMeshDataSet
    from Dataset.timeseries import TimeSeries
    class Mesh(object):
        def __init__(self):
            self.point_sets = {'inlet': np.linspace(0, 100, 100).reshape(-1, 1),
                               'outlet': np.linspace(0, 100, 100).reshape(-1, 1)}
            self.field_sets = {'inlet': np.linspace(0, 100, 100).reshape(-1, 1),
                               'outlet': np.linspace(0, 100, 100).reshape(-1, 1)}


    mesh = Mesh()
    mesh_dataset = SpaceMeshDataSet(mesh, time_interval=None)
    outlet_dataset = mesh_dataset('outlet')

    DataLoaders = DataLoadersManager(datasets={'inlet': mesh_dataset('inlet'),
                                               'outlet': mesh_dataset('outlet')},
                                     batch_sizes={'inlet': 10, 'outlet': 20}, random_seed=2023)
    for epoch in range(2):
        print(epoch)
        for batch in DataLoaders:
            for key, value in batch.items():
                print(key, value[0])

    # mesh_dataloader = SpaceMeshDataLoader(mesh_dataset('outlet'), batch_size=200)



