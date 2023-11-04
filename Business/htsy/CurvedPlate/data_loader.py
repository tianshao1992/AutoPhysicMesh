#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
# @Copyright (c) 2023 Baidu.com, Inc. All Rights Reserved
# @Time    : 2023/10/18 11:08
# @Author  : Liu Tianyuan (liutianyuan02@baidu.com)
# @File    : data_loader.py
# @Description    : ******
"""

import os
import numpy as np

from Utilizes.commons import default
from Dataset.data.mesh import MeshData
from Dataset.imagefield import ImageFieldDataSet
from Dataset.dataloader import DataLoadersManager

n_s = 599
n_x = 9
n_y = 12
n_z = 9
n_t = 180
d_t = 10

def rawdata_preprocess():
    r"""
        coords and temper preprocess
    """
    # 原始数据读
    MeshFile = os.path.join('../data', 'CurvedPlate', 'Job-load-pod-3D.inp')
    mesh = MeshData(MeshFile)
    # mesh.save_file(os.path.join('./data', 'CurvedPlate', 'mesh.msh'), file_type='gmsh')

    # temper = pd.read_csv("./data/CurvedPlate/result.csv", header=None)
    # temper = np.array(temper, dtype=np.float32).reshape(temper.shape[0], -1, n_t).transpose((1, 0, 2))
    # np.save('./data/CurvedPlate/temper_preprocess.npy', temper)

    # 节点排序, x-y方向转换为极坐标
    struct_index = np.lexsort((mesh.points[:, 1], mesh.points[:, 0], mesh.points[:, 2]))
    coords = mesh.points[struct_index].reshape(n_z, n_x, n_y, 3).transpose(1, 2, 0, 3)
    struct_index = struct_index.reshape(n_z, n_x, n_y).transpose(1, 2, 0)
    # y_list_index = np.zeros((n_x, n_y, n_z), dtype=np.int64)
    for k in range(n_z):
        for i in range(n_x):
            y_list_index = coords[i, :, k, 1].argsort()
            struct_index[i, :, k] = struct_index[i, y_list_index, k]
    struct_index = struct_index.reshape(-1)
    # struct_index_ = np.arange(len(struct_index))[struct_index]
    struct_index_ = struct_index.argsort()
    coords = mesh.points[struct_index].astype(np.float32)

    measurements_index = np.loadtxt('../data/CurvedPlate/measurements.txt', dtype=np.int32) - 1
    measurements_index = struct_index_[measurements_index]
    import matplotlib.pyplot as plt
    plt.figure(figsize=(10, 10))
    ax = plt.axes(projection='3d')
    ax.scatter3D(coords[:, 0], coords[:, 1], coords[:, 2], s=0.5)
    ax.scatter3D(coords[measurements_index, 0], coords[measurements_index, 1], coords[measurements_index, 2], s=10)
    plt.show()
    plt.savefig('../data/CurvedPlate/coords.jpg')

    coords = coords.reshape(n_x, n_y, n_z, 3)
    plt.figure(figsize=(10, 10))
    plt.subplot(311)
    # plt.scatter(coords[:, :, 0, 0], coords[:, :, 0, 1])
    for i in range(n_x):
        plt.plot(coords[i, :, 0, 0], coords[i, :, 0, 1])
    for j in range(n_y):
        plt.plot(coords[:, j, 0, 0], coords[:, j, 0, 1])
    plt.subplot(312)
    # plt.scatter(coords[:, 2, :, 0], coords[:, 2, :, 2])
    for i in range(n_z):
        plt.plot(coords[:, 2, i, 0], coords[:, 2, i, 2])
    for j in range(n_x):
        plt.plot(coords[j, 2, :, 0], coords[j, 2, :, 2])
    plt.subplot(313)
    # plt.scatter(coords[5, :, :, 1], coords[5, :, :, 2])
    for i in range(n_y):
        plt.plot(coords[5, i, :, 1], coords[5, i, :, 2])
    for j in range(n_z):
        plt.plot(coords[5, :, j, 1], coords[5, :, j, 2])
    plt.show()
    plt.savefig('../data/CurvedPlate/meshes.jpg')

    # temper_preprocess(n_t)
    temper = np.load('../data/CurvedPlate/temper_preprocess.npy')
    temper = temper[:, struct_index, :].reshape(-1, n_x, n_y, n_z, n_t, 1)

    import matplotlib.pyplot as plt
    # plt.subplot(121)
    plt.figure(figsize=(10, 10))
    plt.subplot(311)
    plt.contourf(coords[:, :, 2, 0],
                   coords[:, :, 2, 1],
                   temper[0, :, :, 2, 100, 0], levels=100, cmap='jet')
    plt.colorbar()
    plt.subplot(312)
    plt.contourf(coords[:, 10, :, 0],
                   coords[:, 10, :, 2],
                   temper[0, :, 10, :, 100, 0], levels=100, cmap='jet')
    plt.colorbar()
    plt.subplot(313)
    plt.contourf(coords[5, :, :, 1],
                   coords[5, :, :, 2],
                   temper[0, 5, :, :, 100, 0], levels=100, cmap='jet')
    plt.colorbar()
    plt.show()
    plt.savefig('../data/CurvedPlate/temper.jpg')

    return coords, temper, measurements_index

from torch.utils.data import Dataset
class TimeSeriesImageFields(Dataset):

    def __init__(self, fields, input_len, output_len,
                 skip_len=None, mask=None):

        self.fields = fields
        self.case_len = fields.shape[0]
        self.time_len = fields.shape[1]
        self.input_len = input_len
        self.output_len = output_len
        self.skip_len = default(skip_len, 0)
        self.mask = default(mask, 1.0)

        self.input_data = self.fields
        self.output_data = self.fields

        self._time_len = (self.time_len - self.input_len - self.output_len - self.skip_len + 1)

    def __getitem__(self, index):

        case_index = index // self._time_len
        time_step = index % self._time_len

        input = self.fields[case_index, time_step:time_step+self.input_len] * self.mask
        output = self.fields[case_index, time_step+self.input_len+self.skip_len:
                                         time_step+self.input_len+self.skip_len+self.output_len]

        return {'input': input, 'target': output}

    def __len__(self):
        return self.case_len * self._time_len

def get_dataloader(config):

    coords, temper, measurements_index = rawdata_preprocess()
    measurements_mask = np.zeros_like(coords[..., 0]).reshape(-1)
    measurements_mask[measurements_index] = 1.0
    measurements_mask = measurements_mask.reshape(n_x, n_y, n_z, 1)
    temper = temper.transpose(0, 4, 1, 2, 3, 5)

    # dataloader 制备
    n_train = config.Training.train_size
    n_valid = config.Training.valid_size

    coords_dataset = ImageFieldDataSet(raw_data={'coords': coords[None]}, input_name='coords', output_name=None,)

    train_dataset = TimeSeriesImageFields(fields=temper[:n_train],
                                          input_len=config.physics.input_len,
                                          output_len=config.physics.output_len,
                                          skip_len=config.physics.skip_len,
                                          mask=measurements_mask)

    train_loaders = DataLoadersManager(datasets={'reconstruct': train_dataset, 'coords': coords_dataset},
                                       batch_sizes=config.Training.train_batch_size,
                                       input_transforms={'reconstruct': 'min-max', 'coords': 'min-max'},
                                       output_transforms={'reconstruct': 'min-max', 'coords': None},
                                       random_seed=config.Seed, shuffle=True)

    input_transforms = train_loaders.input_transforms
    output_transforms = train_loaders.output_transforms

    valid_dataset = TimeSeriesImageFields(
                                          fields=temper[n_train:n_train+n_valid],
                                          input_len=config.physics.input_len,
                                          output_len=config.physics.output_len,
                                          skip_len=config.physics.skip_len,
                                          mask=measurements_mask)

    valid_loaders = DataLoadersManager(datasets={'reconstruct': valid_dataset, 'coords': coords_dataset},
                                       batch_sizes=len(valid_dataset)//n_valid,
                                       input_transforms=input_transforms,
                                       output_transforms=output_transforms,
                                       random_seed=config.Seed, shuffle=False)

    test_dataset = TimeSeriesImageFields(
                                         fields=temper[n_train+n_valid:],
                                         input_len=config.physics.input_len,
                                         output_len=config.physics.output_len,
                                         skip_len=config.physics.skip_len,
                                         mask=measurements_mask)

    test_loaders = DataLoadersManager(datasets={'reconstruct': test_dataset, 'coords': coords_dataset},
                                      batch_sizes=len(test_dataset)//(len(temper)-n_train-n_valid),
                                      input_transforms=input_transforms,
                                      output_transforms=output_transforms,
                                      random_seed=config.Seed, shuffle=False)

    return train_loaders, valid_loaders, test_loaders


if __name__ == "__main__":
    from Business.htsy.CurvedPlate.config.default import get_config
    config = get_config()
    train_loaders, valid_loaders, test_loaders = get_dataloader(config)