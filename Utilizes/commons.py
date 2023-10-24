#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
# @Copyright (c) 2023 Baidu.com, Inc. All Rights Reserved
# @Time    : 2023/10/18 20:46
# @Author  : Liu Tianyuan (liutianyuan02@baidu.com)
# @File    : commons.py
# @Description    : ******
"""
import numpy as np
import PIL.Image as Image

def default(value, d):
    """
        helper taken from https://github.com/lucidrains/linear-attention-transformer
    """
    return d if value is None else value

def identity(x):
    """
    to be used as default activation or function in general
    :param x: tensor or numpy array
    :return: x
    """
    return x

def fig2data(fig):
    """
    fig = plt.figure()
    image = fig2data(fig)
    @brief Convert a Matplotlib figure to a 4D numpy array with RGBA channels and return it
    @param fig a matplotlib figure
    @return a numpy 3D array of RGBA values
    """
    # draw the renderer
    fig.canvas.draw()

    # Get the RGBA buffer from the figure
    w, h = fig.canvas.get_width_height()
    buf = np.frombuffer(fig.canvas.tostring_argb(), dtype=np.uint8)
    buf.shape = (w, h, 4)

    # canvas.tostring_argb give pixmap in ARGB mode. Roll the ALPHA channel to have it in RGBA mode
    buf = np.roll(buf, 3, axis=2)
    image = Image.frombytes("RGBA", (w, h), buf.tobytes())
    image = np.asarray(image)
    return image