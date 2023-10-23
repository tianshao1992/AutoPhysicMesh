#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
# @Copyright (c) 2023 Baidu.com, Inc. All Rights Reserved
# @Time    : 2023/10/22 16:31
# @Author  : Liu Tianyuan (liutianyuan02@baidu.com)
# @File    : operators.py
# @Description    : ******
"""

from Module import bkd, nn
import torch.nn.functional as F

def complex_mul1d(input, weights, use_complex=True):
    # (batch, in_channel, x ), (in_channel, out_channel, x) -> (batch, out_channel, x)
    if use_complex:
        return bkd.einsum("bix, iox->box", input, weights)
    else:
        _real = bkd.einsum("bix, iox->box", input[..., 0], weights[..., 0]) - \
                bkd.einsum("bix, iox->box", input[..., 1], weights[..., 1])
        _imag = bkd.einsum("bix, iox->box", input[..., 0], weights[..., 1]) + \
                bkd.einsum("bix, iox->box", input[..., 1], weights[..., 0])
        return bkd.stack((_real, _imag), dim=-1)

def complex_mul2d(input, weights, use_complex=True):
    # (batch, in_channel, x,y ), (in_channel, out_channel, x,y) -> (batch, out_channel, x,y)
    if use_complex:
        return bkd.einsum("bixy, ioxy->boxy", input, weights)
    else:
        _real = bkd.einsum("bixy, ioxy->boxy", input[..., 0], weights[..., 0]) - \
                bkd.einsum("bixy, ioxy->boxy", input[..., 1], weights[..., 1])
        _imag = bkd.einsum("bixy, ioxy->boxy", input[..., 0], weights[..., 1]) + \
                bkd.einsum("bixy, ioxy->boxy", input[..., 1], weights[..., 0])
        return bkd.stack((_real, _imag), dim=-1)


def complex_mul3d(input, weights, use_complex=True):
    # (batch, in_channel, x,y,t ), (in_channel, out_channel, x,y,t) -> (batch, out_channel, x,y,t)
    if use_complex:
        return bkd.einsum("bixyz,ioxyz->boxyz", input, weights)
    else:
        _real = bkd.einsum("bixyz,ioxyz->boxyz", input[..., 0], weights[..., 0]) - \
                bkd.einsum("bixyz,ioxyz->boxyz", input[..., 1], weights[..., 1])
        _imag = bkd.einsum("bixyz,ioxyz->boxyz", input[..., 0], weights[..., 1]) + \
                bkd.einsum("bixyz,ioxyz->boxyz", input[..., 1], weights[..., 0])
        return bkd.stack((_real, _imag), dim=-1)



def channel_permute(x, forward=True):
    """
    permute the channel dimension
    :param x: tensor
    :param mode: bool
    :return: x: tensor
    """
    if forward:
        if len(x.shape) == 3:
            x = x.permute(0, 2, 1)
        elif len(x.shape) == 4:
            x = x.permute(0, 3, 1, 2)
        elif len(x.shape) == 5:
            x = x.permute(0, 4, 1, 2, 3)
    else:
        if len(x.shape) == 3:
            x = x.permute(0, 2, 1)
        elif len(x.shape) == 4:
            x = x.permute(0, 2, 3, 1)
        elif len(x.shape) == 5:
            x = x.permute(0, 2, 3, 4, 1)
    return x

def spatial_padding(x, padding, forward=True):
    """
    padding the spatial dimension
    :param x:
    :param padding:
    :param forward:
    :return:
    """

    if forward:
        if len(x.shape) == 3:
            x = F.pad(x, [0, padding[0]])
        elif len(x.shape) == 4:
            x = F.pad(x, [0, padding[0], 0, padding[1]])
        elif len(x.shape) == 5:
            x = F.pad(x, [0, padding[0], 0, padding[1], 0, padding[2]])
    else:
        if len(x.shape) == 3:
            x = x[..., :padding[0]]
        elif len(x.shape) == 4:
            x = x[..., :padding[0], :padding[1]]
        elif len(x.shape) == 5:
            x = x[..., :padding[0], :padding[1], :padding[2]]
    return x
