#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
# @Copyright (c) 2023 Baidu.com, Inc. All Rights Reserved
# @Time    : 2023/10/23 0:17
# @Author  : Liu Tianyuan (liutianyuan02@baidu.com)
# @File    : normalizations.py
# @Description    : ******
"""

import numpy as np
import os

from Module import bkd

__all__ = ['DataNormer', 'DimensionNormer', 'UnitNormer']

class DataNormer(object):
    r"""
        data normalization at any dimension
        Args:
            data: ndarray or str
            method: str, "min-max" or "mean-std" or None
            axis: int or tuple, default None = tuple(range(len(data.shape) - 1))
            eps: float, default 1e-10, to avoid zero division
    """
    def __init__(self, data, method="min-max", axis=None, eps=1e-10):

        self.eps = eps

        if isinstance(data, str):
            if os.path.isfile(data):
                try:
                    self.load_file(data)
                except:
                    raise ValueError("the savefile format is not supported!")
            else:
                raise ValueError("the data type is not supported!")

        elif type(data) is np.ndarray:
            if axis is None:
                axis = tuple(range(len(data.shape) - 1))
                self.axis = axis
            self.method = method
            if method == "min-max":
                self.max = np.max(data, axis=axis)
                self.min = np.min(data, axis=axis)
            elif method == "mean-std":
                self.mean = np.mean(data, axis=axis)
                self.std = np.std(data, axis=axis)
            elif method is None:
                pass
            else:
                raise ValueError("the method is not supported!")
        elif data is None:
            self.method = None
        else:
            raise NotImplementedError("the data type is not supported!")

    def norm(self, x):
        """
            input tensors or ndarray
        """
        if bkd.is_tensor(x):
            if self.method == "min-max":
                x = 2 * (x - bkd.tensor(self.min, device=x.device)) \
                    / (bkd.tensor(self.max, device=x.device) - bkd.tensor(self.min, device=x.device) + self.eps) - 1
            elif self.method == "mean-std":
                x = (x - bkd.tensor(self.mean, device=x.device)) / (bkd.tensor(self.std + 1e-10, device=x.device))
            elif self.method == None:
                x = x
        else:
            if self.method == "min-max":
                x = 2 * (x - self.min) / (self.max - self.min + 1e-10) - 1
            elif self.method == "mean-std":
                x = (x - self.mean) / (self.std + 1e-10)
            elif self.method is None:
                x = x
        return x

    def back(self, x):
        """
            input tensors or ndarray
        """
        if bkd.is_tensor(x):
            if self.method == "min-max":
                x = (x + 1) / 2 * (bkd.tensor(self.max, device=x.device)
                                   - bkd.tensor(self.min, device=x.device) + self.eps) + bkd.tensor(self.min,
                                                                                                     device=x.device)
            elif self.method == "mean-std":
                x = x * (bkd.tensor(self.std + 1e-10, device=x.device)) + bkd.tensor(self.mean, device=x.device)
            elif self.method is None:
                x = x
        else:
            if self.method == "min-max":
                x = (x + 1) / 2 * (self.max - self.min + 1e-10) + self.min
            elif self.method == "mean-std":
                x = x * (self.std + 1e-10) + self.mean
            elif self.method is None:
                x = x
        return x

    def save_file(self, save_path):
        import pickle
        with open(save_path, 'wb') as f:
            pickle.dump(self, f)

    def load_file(self, save_path):
        import pickle
        isExist = os.path.exists(save_path)
        if isExist:
            with open(save_path, 'rb') as f:
                b = f.read()
                load = pickle.loads(b)
            self.method = load.method
            if load.method == "mean-std":
                self.std = load.std
                self.mean = load.mean
                self.eps = load.eps
                self.axis = load.axis
            elif load.method == "min-max":
                self.min = load.min
                self.max = load.max
                self.eps = load.eps
                self.axis = load.axis
        else:
            print("The pkl file is not exist, CHECK PLEASE!")


class DimensionNormer(object):

    r"""
        nodimension for physics quantity normalization at any dimension
    """

    def __init__(self, data, axis=None):
        pass
    def norm(self, x):
        """
            input tensors or ndarray
        """
        return x

    def back(self, x):
        """
            input tensors or ndarray
        """
        return x

class UnitNormer(object):

    r"""
        unit normalization at last dimension

    """

    def __init__(self):
        pass

    def norm(self, x):
        """
            input tensors or ndarray
        """
        return x

    def back(self, x):
        """
            input tensors or ndarray
        """
        return x
