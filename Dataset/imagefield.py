#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
# @Copyright (c) 2023 Baidu.com, Inc. All Rights Reserved
# @Time    : 2023/08/22 23:05
# @Author  : Liu Tianyuan (liutianyuan02@baidu.com)
# @File    : ImageField.py
# @Description    : ******
"""

from torch.utils.data import Dataset # todo: the Dataset should be replaced by general bkd.Dataset

from Dataset._base_dataset import BasicDataset
from Utilizes.commons import default

class ImageFieldDataSet(BasicDataset, Dataset):

    def __init__(self,
                 raw_data,
                 input_name: str = None,
                 output_name: str = None,
                 subset_name: str = None,
                 train_mode: bool = True,
                 time_interval: list or tuple = None,
                 time_slices: int = 1,
                 *args, **kwargs):

        super(ImageFieldDataSet, self).__init__(
              raw_data=raw_data,
              input_name=input_name,
              output_name=output_name,
              subset_name=subset_name,
              train_mode=train_mode,
              time_interval=time_interval,
              time_slices=time_slices)

        self.check_data(self.raw_data)

    def __getitem__(self, idx):
        r"""
            get the item of the dataset
        """
        data_dict = {'input': self.input_data[idx]}
        if self.output_data is not None:
            data_dict['target'] = self.output_data[idx]
        return data_dict

    def load_file(self, file):
        r"""
            load the object from a file
            Any non-abstract dataset inherited from this class should implement this method.
            Args:
                file(str): The path of the file to save to
            Returns:
                None
        """
        pass
        raise NotImplementedError("load_file not implemented yet!")

    def save_file(self, file):
        r"""
            save the object to a file
            Any non-abstract dataset inherited from this class should implement this method.
            Args:
                file(str): The path of the file to save to
            Returns:
                None
        """
        pass
        raise NotImplementedError("save_file not implemented yet!")


    def train_test_split(self, train_size: float = 0.8, shuffle: bool = True, random_seed: int = 2023):

        pass

        raise NotImplementedError("train_test_split not implemented yet!")


    def check_data(self, raw_data):
        r"""
            check the data
            :param raw_data:
            :return:
        """
        len_data = None
        for key in raw_data.keys():
            len_data = len(raw_data[key]) if len_data is None else len_data
            if len_data != len(raw_data[key]):
                raise ValueError("raw_data length not equal!")

    def __call__(self,
                 input_name: str,
                 output_name: str,
                 subset_name: str = None,
                 train_mode: bool = None,
                 time_slices: int = 1,
                 time_interval: list or tuple = None,
                 *args, **kwargs):

        return ImageFieldDataSet(raw_data=self.raw_data,
                                 input_name=input_name,
                                 output_name=output_name,
                                 subset_name=subset_name,
                                 train_mode=default(train_mode, self.train_mode),
                                 time_slices=default(time_slices, self.time_slices),
                                 time_interval=default(time_interval, self.time_interval),)

    def __len__(self):
        return len(self.input_data)

if __name__ == '__main__':

    import numpy as np

    raw_data = {'input': np.random.randn(100, 2),
                'output': np.random.randn(100, 100, 3),
                'target': np.random.randn(100, 3)}

    field_dataset = ImageFieldDataSet(raw_data, input_name='input', output_name='output')

