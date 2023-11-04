#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
# @Copyright (c) 2023 Baidu.com, Inc. All Rights Reserved
# @Time    : 2023/10/30 15:49
# @Author  : Liu Tianyuan (liutianyuan02@baidu.com)
# @File    : params.py
# @Description    : ******
"""
import os
from typing import Callable, List, Union
from ml_collections import ConfigDict
from scipy.io import loadmat, savemat
import numpy as np
import pandas as pd
from Utilizes.commons import default
from Dataset.data._base import BasicData


class MetaParams(object):
    def __init__(self, name, unit, mode, type, data,
                 lb=None, ub=None,
                 description=None, *args, **kwargs):
        r"""
           meta  params
           Args:
                name: name of the params, str
                unit: unit of the params, str or list
                mode: mode of the params, str 'constant' or 'variable'
                type: type of the params, 'int', 'float', 'str', 'list', 'tuple', 'dict', 'array'
                # todo: add more types for fields, mesh, scalars and series.
                data: data of the params, numpy array or list or bkd.tensor
                # todo: add the lowbound and upbound to the params.
                lb: low bound of the params, list or float or int, default is -np.inf
                ub: up bound of the params, list or float or int, default is np.inf
                description: description of the params, str
        """
        self.name = name
        self.unit = unit
        self.mode = mode
        self.data = data
        self.type = type
        self.lb = default(lb, -np.inf)
        self.ub = default(ub, np.inf)
        self.description = default(description, name)

        self._raw = data  # the initial value of the data

    #     todo: add more methods and properties

    @property
    def value(self):
        return self.data

    @property
    def init(self):
        return self._raw

class ParamsData(BasicData):

    def __init__(self,
                 data: Union[str, dict],
                 mode: Union[tuple, list, dict, str],
                 unit: Union[tuple, list] = None,
                 description: Union[tuple, list] = None,
                 *args, **kwargs):

        super(ParamsData, self).__init__()
        self.data_setup(data, mode)

    def data_setup(self, data, mode):

        if isinstance(data, str) and os.path.isfile(data):
            _data = self.load_file(data)
        elif isinstance(data, dict):
            _data = data
        else:
            raise NotImplementedError("the data type is not supported!")

        if isinstance(mode, str):
            mode = [mode] * len(_data.keys())
        elif isinstance(mode, (list, tuple)):
            assert len(mode) == len(_data.keys()), "the length of mode must be equal to the length of data"
        elif isinstance(mode, dict):
            assert mode.keys() == _data.keys(), "the keys of mode must be equal to the keys of data"
        else:
            raise NotImplementedError("the mode type is not supported!")

        mode = dict(zip(_data.keys(), mode))
        self._data = _data
        self.names = []
        for name in _data.keys():
            assert mode[name] in ['constant', 'variable'], "the mode must be 'constant' or 'identify'!"
            # note that parameter names are unique.
            assert name not in self.names, "the parameter name must be unique!"
            setattr(self, name, MetaParams(mode=mode[name],
                                           name=name,
                                           type=type(_data[name]),
                                           data=_data[name],
                                           unit=None,  # add unit later
                                           description=None,  # add description later
                                           ))
            self.names.append(name)


if __name__ == "__main__":
    parmas = ParamsData(data={'a': 1, 'b': 2.}, mode={'a': 'constant', 'b': 'variable'})

    a = 0
