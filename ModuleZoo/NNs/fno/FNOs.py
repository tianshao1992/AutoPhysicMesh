#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
# @Copyright (c) 2022 Baidu.com, Inc. All Rights Reserved
# @Time    : 2022/11/26 2:14
# @Author  : Liu Tianyuan (liutianyuan02@baidu.com)
# @Site    : 
# @File    : FNOs.py
"""
import os
import sys

# add configs.py path
file_path = os.path.abspath(os.path.dirname(__file__))
sys.path.append(os.path.join(file_path.split('fno')[0]))
sys.path.append(os.path.join(file_path.split('Models')[0]))

from Module import bkd, nn
from Module.NNs.activations import get as get_activation
from ModuleZoo.NNs.fno.spectral_layers import SpectralConv1d, SpectralConv2d, SpectralConv3d
from ModuleZoo.NNs.fno.basic_operators import channel_permute, spatial_padding

class FNO(nn.Module):
    """
        1维FNO网络
    """

    def __init__(self,
                 input_dim: int,
                 output_dim: int,
                 spatial_dim: int,
                 spectral_modes: int or list or tuple,
                 spectral_norm: str = None,
                 layer_width: int = 64,
                 layer_depth: int = 4,
                 layer_active: str = 'gelu',
                 last_width: int = 128,
                 time_steps: int = 1,
                 spatial_padding: int or list or tuple = 0,
                 use_complex=True,
                 *args, **kwargs):
        super(FNO, self).__init__()
        """
        The overall network. It contains /depth/ layers of the Fourier layer.
        1. Lift the input to the desire channel dimension by self.fc0 .
        2. /depth/ layers of the integral operators u' = (W + K)(u).
            W defined by self.w; K defined by self.conv .
        3. Project from the channel space to the output space by self.fc1 and self.fc2 .

        input: the solution of the initial condition and location (a(x), x)
        input shape: (batchsize, x=s, c=2)
        output: the solution of a later timestep
        output shape: (batchsize, x=s, c=1)
        """
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.spatial_dim = spatial_dim

        if isinstance(spectral_modes, int):
            spectral_modes = [spectral_modes] * self.spatial_dim
        assert len(spectral_modes) == self.spatial_dim, (
            "the length of spectral_modes must be {}, but got {}".format(self.spatial_dim, len(spectral_modes)))
        self.spectral_modes = spectral_modes

        self.layer_width = layer_width
        self.layer_depth = layer_depth
        self.layer_active = get_activation(layer_active)
        self.last_width = last_width

        self.time_steps = time_steps
        self.use_complex = use_complex

        if isinstance(spatial_padding, int):
            spatial_padding = [spatial_padding] * self.spatial_dim
        assert len(spatial_padding) == self.spatial_dim, (
            "the length of spatial_padding must be {}, but got {}".format(self.spatial_dim, len(spatial_padding)))
        self.spatial_padding = spatial_padding  # pad the domain if input is non-periodic
        self._spatial_padding = []
        for i in range(self.spatial_dim):
            self._spatial_padding.append(-self.spatial_padding[i] if self.spatial_padding[i] != 0 else None)


        self.fc0 = nn.Linear(self.time_steps * self.input_dim + self.spatial_dim, self.layer_width)
        self.fc1 = nn.Linear(self.layer_width, self.last_width)
        self.fc2 = nn.Linear(self.last_width, self.output_dim)

        self.spectral_convs = nn.ModuleList()
        for i in range(self.layer_depth):
            if self.spatial_dim == 1:
                self.spectral_convs.append(SpectralConv1d(self.layer_width, self.layer_width, self.spectral_modes,
                                                          layer_active=self.layer_active,
                                                          spectral_norm=spectral_norm,
                                                          use_complex=self.use_complex))
            elif self.spatial_dim == 2:
                self.spectral_convs.append(SpectralConv2d(self.layer_width, self.layer_width, self.spectral_modes,
                                                          layer_active=self.layer_active,
                                                          spectral_norm=spectral_norm,
                                                          use_complex=self.use_complex))
            elif self.spatial_dim == 3:
                self.spectral_convs.append(SpectralConv3d(self.layer_width, self.layer_width, self.spectral_modes,
                                                          layer_active=self.layer_active,
                                                          spectral_norm=spectral_norm,
                                                          use_complex=self.use_complex))
            else:
                raise ValueError("spatial_dim must be 1, 2 or 3, but got {}".format(self.space_dim))


    def forward(self, x, grid):
        """
        forward computation
        """
        # x dim = [b, x1, t*v]
        x = bkd.cat((x, grid), dim=-1)
        x = self.fc0(x)

        assert self.spatial_dim == len(x.shape) - 2, (
            "the spatial dim must be {}, but got {}".format(len(x.shape) - 2, self.spatial_dim))

        if self.spatial_dim >= 1 and self.spatial_dim <=3:
            x = channel_permute(x, forward=True)
            x = spatial_padding(x, self.spatial_padding, forward=True)
        else:
            raise ValueError("the spatial dim or tensor must be 1, 2 or 3, "
                             "but got {}".format(self.spatial_dim))

        for i in range(self.layer_depth):
            x = self.spectral_convs[i](x)

        if self.spatial_dim >= 1 and self.spatial_dim <=3:
            x = spatial_padding(x, self._spatial_padding, forward=False)
            x = channel_permute(x, forward=False)
        else:
            raise ValueError("the spatial dim or tensor must be 1, 2 or 3, "
                             "but got {}".format(self.spatial_dim))

        x = self.fc1(x)
        x = self.layer_active(x)
        x = self.fc2(x)
        return x


if __name__ == '__main__':
    x = bkd.ones([10, 32, 4])
    g = bkd.ones([10, 32, 1])
    layer = FNO(input_dim=4,
                output_dim=1,
                spatial_dim=1,
                spectral_modes=16,
                layer_width=64,
                layer_depth=4,
                time_steps=1,
                layer_active='gelu')

    y = layer(x, g)
    print(y.shape)

    x = bkd.ones([10, 9, 10, 4])
    g = bkd.ones([10, 9, 10, 2])
    layer = FNO(input_dim=4,
                output_dim=2,
                spatial_dim=2,
                spectral_modes=4,
                layer_width=64,
                layer_depth=4,
                time_steps=1,
                layer_active='gelu')

    y = layer(x, g)
    print(y.shape)

    x = bkd.ones([10, 32, 32, 32, 4])
    g = bkd.ones([10, 32, 32, 32, 3])

    layer = FNO(input_dim=4,
                output_dim=3,
                spatial_dim=3,
                spectral_modes=16,
                layer_width=64,
                layer_depth=4,
                time_steps=1,
                spatial_padding=2,
                layer_active='gelu')

    y = layer(x, g)
    print(y.shape)