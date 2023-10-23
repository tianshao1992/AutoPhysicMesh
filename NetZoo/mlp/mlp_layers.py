#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
# @Copyright (c) 2023 Baidu.com, Inc. All Rights Reserved
# @Time    : 2022/10/22 22:52
# @Author  : Liu Tianyuan (liutianyuan02@baidu.com)
# @File    : mlp_layers.py
# @Description    : ******
"""

from Module import bkd, nn
from Module.activations import get as get_activation

class MlpBlock(nn.Module):
    def __init__(self, planes: list, active="gelu",
                 last_active=False,
                 use_one_branch=True):
        # =============================================================================
        #     Inspired by Haghighat Ehsan, et all.
        #     "A physics-informed deep learning framework for inversion and surrogate modeling in solid mechanics"
        #     Computer Methods in Applied Mechanics and Engineering.
        # =============================================================================
        super(MlpBlock, self).__init__()
        self.planes = planes
        self.active = get_activation(active)
        self.last_active = last_active
        self.use_one_branch = use_one_branch

        self.layers = nn.ModuleList()
        if self.use_one_branch:
            for i in range(len(self.planes) - 1):
                self.layers.append(nn.Linear(self.planes[i], self.planes[i + 1]))
            self.layers = nn.Sequential(*self.layers)
        else:
            for j in range(self.planes[-1]):
                layer = []
                for i in range(len(self.planes) - 2):
                    layer.append(nn.Linear(self.planes[i], self.planes[i + 1]))
                layer.append(nn.Linear(self.planes[-2], 1))
                self.layers.append(nn.Sequential(*layer))

        self.reset_parameters()

    def reset_parameters(self):
        """
        weight initialize
        """
        for m in self.modules():
            if isinstance(m, nn.Linear):
                # nn.init.xavier_normal_(m.weight, gain=1)
                nn.init.xavier_uniform_(m.weight, gain=1)
                m.bias.data.zero_()

    def forward(self, x, u=None):
        """
        forward compute
        :param in_var: (batch_size, ..., input_dim)
        """
        if u is not None:
            assert isinstance(u, (list, tuple)) and len(u) == 2, "u must be a list of two tensors"

        if self.use_one_branch:
            for layer in self.layers[:-1]:
                x = self.active(layer(x))
                if u is not None:
                    assert u[0].shape == x.shape and u[1].shape == x.shape, "u and x must have the same shape"
                    x = x * u[0] + (1 - x) * u[1]
            y = self.layers[-1](x)
            if self.last_active:
                y = self.active(y)
            return y
        else:
            ys = []
            for i in range(self.planes[-1]):
                for i, layer in enumerate(self.layers[i][:-1]):
                    if i == 0:
                        y = self.active(layer(x))
                    else:
                        y = self.active(layer(y))
                    if u is not None:
                        y = y * u[0] + (1 - y) * u[1]
                y = self.layers[i][-1](y)
                if self.last_active:
                    y = self.active(y)
                ys.append(y)
            return bkd.cat(ys, dim=-1)