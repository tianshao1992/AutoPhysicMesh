#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
# @Copyright (c) 2023 Baidu.com, Inc. All Rights Reserved
# @Time    : 2023/10/16 12:50
# @Author  : Liu Tianyuan (liutianyuan02@baidu.com)
# @File    : net_model.py
# @Description    : ******
"""

from model import bkd, nn
from model import activations

class FourierEmbedding(nn.Module):

    def __init__(self, input_dim, hidden_dim, output_dim, scale=1.0, modes=1):
        super(FourierEmbedding, self).__init__()
        self.scale = scale
        self.modes = modes + 1
        self.hidden_dim = hidden_dim
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.fourier_weight = nn.Parameter(self.scale * bkd.rand(input_dim, hidden_dim * self.modes, dtype=bkd.float32))
        self.linear = nn.Linear(hidden_dim * 2 * self.modes, output_dim)
        self.register_buffer(name='modes_harmonic', tensor=bkd.arange(0, self.modes, 1))
    def forward(self, x):
        bsz = list(x.shape[:-1])
        h = bkd.matmul(x, self.fourier_weight)
        h = h.view(bsz + [self.hidden_dim, self.modes])
        h = bkd.cat((bkd.cos(h * self.modes_harmonic),
                     bkd.sin(h * self.modes_harmonic)), dim=-1).view(bsz + [self.hidden_dim * 2 * self.modes])
        y = self.linear(h)
        return y


class MlpNet(nn.Module):

    def __init__(self,
                 input_dim: int,
                 output_dim: int,
                 layer_depth: int,
                 layer_width: int,
                 layer_active: str,
                 modified_mode: bool = False,
                 input_transform: callable = None,
                 output_transform: callable = None,
                 use_one_branch: bool = True,
                 *args, **kwargs):
        super(MlpNet, self).__init__()

        # =============================================================================
        #     Inspired by Wang, Sifan and Sankaran, Shyam and Wang, Hanwen and Perdikaris, Paris.
        #     "Wang, Sifan and Sankaran, Shyam and Wang, Hanwen and Perdikaris, Paris"
        #     arXiv preprint arXiv:2308.08468. 2023
        # =============================================================================

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.layer_depth = layer_depth
        self.layer_width = layer_width
        self.layer_active = activations.get(layer_active)
        self.modified_mode = modified_mode
        self.input_transform = input_transform
        self.output_transform = output_transform
        self.use_one_branch = use_one_branch

        if input_transform is None:
            self.input_transform = MlpBlock(planes=[self.input_dim, self.layer_width],
                                            active=layer_active, last_active=True)

        if modified_mode:
            self.modified_transform = MlpBlock(planes=[self.layer_width, self.layer_width*2],
                                               active=layer_active, last_active=True)

        self.MlpBlock = MlpBlock(planes=[layer_width, ] * layer_depth + [output_dim, ],
                                 active=layer_active,
                                 last_active=False,
                                 use_one_branch=use_one_branch)

    def forward(self, x):

        h = self.input_transform(x)
        if self.modified_mode:
            u = bkd.chunk(self.modified_transform(h), chunks=2, dim=-1)
        else:
            u = None

        y = self.MlpBlock(h, u)

        if self.output_transform is not None:
            y = self.output_transform(y)
        return y


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
        self.active = activations.get(active)
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


if __name__ == "__main__":
    model = MlpBlock([3, 64, 64, 3], active="gelu", last_active=False, use_one_branch=True)
    print(model)
    x = bkd.ones([100, 50, 3])
    y = model(x)
    print(y.shape)

    model = MlpBlock([3, 64, 64, 3], active="gelu", last_active=True, use_one_branch=False)
    print(model)
    x = bkd.ones([100, 50, 3])
    y = model(x)
    print(y.shape)


    model = FourierEmbedding(input_dim=3, hidden_dim=16, output_dim=64, modes=10)
    x = bkd.ones([100, 50, 3])
    y = model(x)
    print(y.shape)


    model = MlpNet(input_dim=3,
                   output_dim=2,
                   layer_depth=5,
                   layer_width=64,
                   layer_active='gelu',
                   modified_mode=True,)
    #
    x = bkd.ones([100, 50, 3])
    y = model(x)
    print(y.shape)

    model = MlpNet(input_dim=3,
                   output_dim=2,
                   layer_depth=5,
                   layer_width=64,
                   layer_active='gelu',
                   modified_mode=True,
                   input_transform=FourierEmbedding(input_dim=3, hidden_dim=4, output_dim=64, scale=1.0))
    #
    x = bkd.ones([100, 50, 3])
    y = model(x)
    print(y.shape)