#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
# @Copyright (c) 2023 Baidu.com, Inc. All Rights Reserved
# @Time    : 2023/10/25 0:02
# @Author  : Liu Tianyuan (liutianyuan02@baidu.com)
# @File    : seriesdata.py
# @Description    : ******
"""

import os
from typing import Callable, List, Union

from scipy.io import loadmat, savemat
import numpy as np
import pandas as pd

from Dataset.data._base_data import BasicData

class SeriesData(BasicData):
    def __init__(self,
                 data: Union[pd.DataFrame, pd.Series, str],
                 freq: Union[int, str],
                 time_col: str = None,
                 value_cols: List[str] = None,
                 *args, **kwargs):

        super(SeriesData, self).__init__()

        if isinstance(data, pd.DataFrame):
            _data = data
        elif isinstance(data, pd.Series):
            _data = data.to_frame()
        elif isinstance(data, str) and os.path.isfile(data):
            suffix = os.path.splitext(data)[-1][1:]
            _data = self.load_file(data, file_type=suffix)
        else:
            raise NotImplementedError("the data type is not supported!")

        self._data = _data
        self.data = self.load_dataframe(_data, freq, time_col, value_cols, *args, **kwargs)
        self.freq = freq

        pass

    def load_file(self, file, file_type='csv'):
        r"""
            load mesh file
            :param file:
            :return:
        """
        assert os.path.exists(file), 'file not found: ' + file
        if file_type == 'csv':
            dataframe = pd.read_csv(file, index_col=0)
        elif file_type == 'mat':
            mat = loadmat(file)
            dataframe = pd.DataFrame(mat['data'], index=mat['index'], columns=mat['cols'])
        elif file_type == 'h5':
            raise NotImplementedError('file type not supported: ' + file_type)
        else:
            raise NotImplementedError('file type not supported: ' + file_type)

        return dataframe

    def save_file(self, file, file_type):
        r"""
            save mesh file
            Args:
                :param file: file path
                :param file_type: 'csv', 'mat'
            Return:
                None
        """
        if file_type == 'csv':
            self.data.to_csv(file)
        elif file_type == 'mat':
            mat = savemat(file,{'data': self.data.values,
                                'index': self.data.index,
                                'cols': self.data.columns})
        elif file_type == 'h5':
            raise NotImplementedError('file type not supported: ' + file_type)
        else:
            raise NotImplementedError('file type not supported: ' + file_type)


    def load_dataframe(self,
                       dataframe: pd.DataFrame,
                       freq: Union[int, str] = None,
                       time_col: str = None,
                       value_cols: List[str] = None,
                       interpolate_method: Union[str, Callable] = 'linear',
                       dtype: np.dtype = None,
                       *args, **kwargs) -> pd.DataFrame:
        r"""
            get point sets
            :return: point sets
        """

        if time_col is None:
            time_val = dataframe.index.copy(deep=True)
        else:
            time_val = dataframe.loc[:, (time_col,)].copy(deep=True)
        time_index = self.time_convert(time_val)

        if value_cols is None:
            value_val = dataframe.loc[:, dataframe.columns != time_col].copy(deep=True)
        else:
            value_val = dataframe.loc[:, value_cols].copy(deep=True)

            # time_index = pd.date_range(start=time_val[0], end=time_val[-1], periods=freq)

        value_val.set_index(time_index, inplace=True)
        value_val.sort_index(inplace=True)

        if freq is None:
            raise ValueError("freq is None!")
        else:
            value_val = value_val.resample(rule=freq).mean()  # 采用nan填充, 保证时间序列长度一致

        value_val.fillna(method='ffill', inplace=True)
        value_val.fillna(method='bfill', inplace=True)
        value_val.interpolate(method=interpolate_method, inplace=True)

        return value_val

    def time_convert(self, time_val) -> pd.DatetimeIndex:
        r"""
            convert time to pd.DatetimeIndex
            :param time_val: time
            :return: pd.DatetimeIndex
        """
        if isinstance(time_val[0], str):
            time_index = pd.to_datetime(time_val)
        elif isinstance(time_val[0], pd.Timestamp):
            time_index = pd.DatetimeIndex(time_val)
        elif isinstance(time_val[0], pd.DatetimeIndex):
            time_index = time_val
        else:
            raise NotImplementedError("time_val type not supported!")

        return time_index

    def visual_matplot(self, idx: int):

        return NotImplementedError("visual_matplot not implemented yet!")



if __name__ == "__main__":


    # use a real mesh class to test
    series = SeriesData('../demo/lzjy1d_t/data/raw_data.csv', freq='1s')

    import matplotlib.pyplot as plt
    plt.plot(outlet_dataset.mesh_points[:, 0], outlet_dataset.mesh_points[:, 1],
                c=outlet_dataset.mesh_fields[:, 0], cmap='jet')
    plt.show()