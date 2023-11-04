#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
# @Copyright (c) 2023 Baidu.com, Inc. All Rights Reserved
# @Time    : 2023/9/6 15:27
# @Author  : Liu Tianyuan (liutianyuan02@baidu.com)
# @File    : geometric.py
# @Description    : ******
"""

import numpy as np

def ccw_sort(points):
    r"""
        Sort given polygon points in CCW order
        Args:
            points: input points data, [N, 2]
        Return:
            new_points:, indexed points data, [N, 2]
    """

    points = np.array(points)
    mean = np.mean(points, axis=0)
    coords = points - mean
    s = np.arctan2(coords[:, 0], coords[:, 1])
    return points[np.argsort(s)]