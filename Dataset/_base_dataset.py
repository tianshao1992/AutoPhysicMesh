#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
# @Copyright (c) 2023 Baidu.com, Inc. All Rights Reserved
# @Time    : 2023/10/24 20:38
# @Author  : Liu Tianyuan (liutianyuan02@baidu.com)
# @File    : _base_dataset.py
# @Description    : ******
"""
import abc
import os
from abc import ABC
from Utilizes.commons import default

class BasicDataset(ABC):
    def __init__(self,
                 raw_data: dict or str,
                 input_name: str = None,
                 output_name: str = None,
                 subset_name: str = None,
                 **kwargs):
        r"""
            BasicDataset is the base class of all datasets
        :param raw_data:
        :param input_name:
        :param output_name:
        :param subset_name:
        :param kwargs:
        """

        self._build(raw_data=raw_data,
                    input_name=input_name,
                    output_name=output_name,
                    subset_name=subset_name,
                    **kwargs)

    def _build(self,
               raw_data,
               input_name: str,
               output_name: str,
               subset_name: str,
               **kwargs):

        r"""
            build the dataset
            Args:
                :param raw_data:
                :param input_name:
                :param output_name:
                :param subset_name:
                :param kwargs:
            Return:
                None
        """
        # assert isinstance(mesh, (meshio.Mesh)), "mesh should be meshio.Mesh or mesh file"

        if isinstance(raw_data, str) and os.path.isfile(raw_data):
            raw_data = self.load_file(raw_data, file_type=None)

        # self.check_data(raw_data)
        self.raw_data = raw_data
        self.input_name = input_name
        self.output_name = output_name
        self.subset_name = subset_name

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

        if subset_name is not None:
            assert subset_name in self.raw_data.keys(), "subset_name does not exist!"
            self.subset_data = self.raw_data[subset_name]
        else:
            self.subset_data = None

        for key, value in kwargs.items():
            setattr(self, key, value)

    @abc.abstractmethod
    def load_file(self, file, file_type, *args, **kwargs):
        r"""
            load the object from a file
            Any non-abstract dataset inherited from this class should implement this method.
            Args:
                file(str): The path of the file to save to
                file_type(str): The type of the file to save to
            Returns:
                None
        """
        return NotImplementedError("base method not implemented")

    def save_file(self, file, file_type, *args, **kwargs):
        r"""
            save the object to a file
            Any non-abstract dataset inherited from this class should implement this method.
            Args:
                file(str): The path of the file to save to
                file_type(str): The type of the file to save to
            Returns:
                None
        """
        return NotImplementedError("base method not implemented")

    def check_data(self, data, *args, **kwargs):
        r"""
            check the data
            Any non-abstract dataset inherited from this class should implement this method.
            Args:
                data: The data to check
        :return:
            None
        """

        return NotImplementedError("base method not implemented")

    def __call__(self, input_name, output_name, subset_name, **kwargs):
        r"""
            call the object
        :param input_name:
        :param output_name:
        :param subset_name:
        :param kwargs:
        :return:
        """
        return self


