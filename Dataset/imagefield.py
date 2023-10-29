#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
# @Copyright (c) 2023 Baidu.com, Inc. All Rights Reserved
# @Time    : 2023/10/22 23:05
# @Author  : Liu Tianyuan (liutianyuan02@baidu.com)
# @File    : ImageField.py
# @Description    : ******
"""


import os
from torch.utils.data import Dataset
from Utilizes.commons import default

from Dataset.dataprocess import DataNormer


class ImageFieldDataSet(Dataset):

    def __init__(self,
                 raw_data,
                 input_name: str,
                 output_name: str,
                 train_mode: bool = True,
                 time_interval: list or tuple = None,
                 time_slices: int = 1,
                 *args, **kwargs):
        super(ImageFieldDataSet, self).__init__(*args, **kwargs)

        self._setup(raw_data=raw_data,
                    input_name=input_name,
                    output_name=output_name,
                    train_mode=train_mode,
                    time_interval=time_interval,
                    time_slices=time_slices)

    def _setup(self,
               raw_data,
               input_name: str,
               output_name: str,
               train_mode: bool,
               time_interval: list or tuple,
               time_slices: int):
        # assert isinstance(mesh, (meshio.Mesh)), "mesh should be meshio.Mesh or mesh file"

        if isinstance(raw_data, str) and os.path.isfile(raw_data):
            raw_data = self.load_file(raw_data)

        self._check_data(raw_data)

        self.raw_data = raw_data
        self.train_mode = train_mode
        self.time_slices = time_slices
        self.time_interval = time_interval

        if input_name is not None:
            assert input_name in self.raw_data.keys(), "input_name does not exist!"
            self.input_data = self.raw_data[input_name]
        else:
            self.input_data = None

        if output_name is not None:
            assert output_name in self.raw_data.keys(), "output_name does not exist!"
            self.output_data = self.raw_data[output_name]
        else:
            self.output_data = None

    def __getitem__(self, idx):
        return {'input': self.input_data[idx],
                'target': self.output_data[idx] if self.output_data is not None else None}


    def load_file(self, file):
        pass
        raise NotImplementedError("load_file not implemented yet!")

    def _check_data(self, raw_data):
        len_data = None
        for key in raw_data.keys():
            len_data = len(raw_data[key]) if len_data is None else len_data
            if len_data != len(raw_data[key]):
                raise ValueError("raw_data length not equal!")
    def __call__(self,
                 input_name: str,
                 output_name: str,
                 train_mode: bool = None,
                 time_slices: int = 1,
                 time_interval: list or tuple = None,
                 *args, **kwargs):
        return ImageFieldDataSet(raw_data=self.raw_data,
                                 input_name=input_name,
                                 output_name=output_name,
                                 train_mode=default(train_mode, self.train_mode),
                                 time_slices=default(time_slices, self.time_slices),
                                 time_interval=default(time_interval, self.time_interval),)

    def __len__(self):
        return len(self.input_data)