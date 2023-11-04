#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
# @Copyright (c) 2023 Baidu.com, Inc. All Rights Reserved
# @Time    : 2023/10/30 13:15
# @Author  : Liu Tianyuan (liutianyuan02@baidu.com)
# @File    : data_loader.py
# @Description    : ******
"""

import numpy as np
import xlrd
import scipy.io as sio
from Dataset.data.params import ParamsData
from Dataset.imagefield import ImageFieldDataSet
from Dataset.dataloader import DataLoadersManager

def load_raw(name, mode='equal', power="small"):
    # power = {'small' , '40w', '22w'}
    # names = {'O-1':0, 'O-4':1, 'F-1':2, 'F-4':3, 'P-1':4, 'P-2':5, 'Si-1':6, 'Si-4':7}

    if mode == "equal":
        excel = xlrd.open_workbook("./data/data_.xls")  # 打开excel文件
        sheet = excel.sheet_by_name(name)  # 获取工作薄

        data = []

        if power == "small":
            index = (1, 2, 3)
        elif power == "40w":
            index = (6, 7, 8)
        elif power == "22w":
            index = (11, 12, 13)

        for i in index:
            rows: list = sheet.col_values(i)  # 获取第一行的表头内容
            vals = []
            for r in rows:
                if r != '':
                    vals.append(r)
                else:
                    break
            data.append(vals)

        data = np.array(data).astype(np.float32).transpose((1, 0))
        property = sheet.col_values(index[-1]+1)[0:7]
        property_names = ['rho', 'cp0', 'lambda0', 'lambda1', 'delta', 'power', 'length']
        property_constant = dict(zip(property_names, property))
        # initial property to identify
        property_constant.update({'q0': property_constant['power']/(property_constant['length']**2 * 2)})
        property_constant.update({'cp0': 1 / property_constant['cp0']})

        property_identify = {}
        property_identify.update({'cp': 1 / 800., 'lambdas': 0.02})
        return data, property_constant, property_identify
    else:
        data = sio.loadmat("data\\" + name + ".mat")
        material = np.array(data['matr'], dtype=np.float32).tolist()
        material.append(data['lamda0'].astype(np.float32).squeeze())
        try:
            material.append(data['alpha'].astype(np.float32).squeeze())
        except:
            pass

        field = np.array(data['sol_pred'], dtype=np.float32).transpose((1, 0))
        field = field[::2, :]

        # alpha_exp = np.concatenate((data['x0'], data['y0']), axis=-1).astype(np.float32)
        # alpha_fit = np.concatenate((data['x1'], data['y1']), axis=0).astype(np.float32).transpose((1, 0))

        data = []
        excel = xlrd.open_workbook("data/data.xls")  # 打开excel文件
        sheet = excel.sheet_by_name(name)  # 获取工作薄
        index = (6, 7, 8)
        for i in index:
            rows: list = sheet.col_values(i)  # 获取第一行的表头内容
            vals = []
            for r in rows:
                if r != '':
                    vals.append(r)
                else:
                    break
            data.append(vals)
        data = np.array(data).astype(np.float32)

        from scipy.interpolate import interp1d
        t = np.linspace(data[0, 0], data[0, -1], 2000)
        f_exp = interp1d(data[0, :], data[1, :], kind='cubic')
        f_ana = interp1d(data[0, :], data[2, :], kind='cubic')
        T_exp = f_exp(t)
        T_ana = f_ana(t)
        data = np.stack((t, T_exp, T_ana), axis=0)

        return data, material, field

def get_dataloader(config):

    material_name = config.physics.material_name
    identify_mode = config.physics.identify_mode
    heating_power = config.physics.heating_power

    # todo: support different identify_modes
    data, property_constant, property_identify = load_raw(name=material_name, mode=identify_mode, power=heating_power)

    n_s = config.physics.time_resolution

    constant_params = ParamsData(property_constant, mode='constant')
    variable_params = ParamsData(property_identify, mode='variable')

    train_dataset = ImageFieldDataSet(raw_data={
        't': data[::n_s, 0].reshape(-1, 1),
        'T_exp': data[::n_s, 1].reshape(-1, 1),
        'T_LMM': data[::n_s, 2].reshape(-1, 1),
    })

    train_loaders = DataLoadersManager(datasets={'exp': train_dataset(input_name='t', output_name='T_exp'),
                                                 'LMM': train_dataset(input_name='t', output_name='T_LMM'),
                                                    },
                                       batch_sizes=config.Training.train_batch_size,
                                       random_seed=config.Seed,
                                       shuffle=True)

    valid_dataset = ImageFieldDataSet(raw_data={
        't': data[:, 0].reshape(-1, 1),
        'T_exp': data[:, 1].reshape(-1, 1),
        'T_LMM': data[:, 2].reshape(-1, 1),
    })

    valid_loaders = DataLoadersManager(datasets={'exp': valid_dataset(input_name='t', output_name='T_exp'),
                                                 'LMM': valid_dataset(input_name='t', output_name='T_LMM'),
                                                    },
                                       batch_sizes=10000,  # to sample all data
                                       random_seed=config.Seed,
                                       shuffle=False)


    return train_loaders, valid_loaders, (constant_params, variable_params)




if __name__ == "__main__":

    from demo.thermalidentify_0d_t.conifg.all_config import get_config
    config = get_config()
    train_loaders, valid_loaders, physics_params = get_dataloader(config)
