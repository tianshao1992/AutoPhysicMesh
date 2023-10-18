#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
# @Copyright (c) 2023 Baidu.com, Inc. All Rights Reserved
# @Time    : 2023/10/17 0:02
# @Author  : Liu Tianyuan (liutianyuan02@baidu.com)
# @File    : lossfuncs.py
# @Description    : ******
"""

__all__ = ['get']

from model import bkd, nn

def mean_absolute_error(y_pred, y_true):
    """
    calculate the mean absolute error between y_true and y_pred
    """
    return bkd.mean(bkd.abs(y_true - y_pred))


def mean_absolute_percentage_error(y_pred, y_true):
    """
    calculate the mean absolute percentage error between y_true and y_pred
    """
    # todo: should we add a small epsilon to avoid dividing by zero and add max calculation?
    return bkd.mean(bkd.abs((y_true - y_pred) / y_true)) * 100


def mean_squared_error(y_pred, y_true):
    """
    calculate the mean squared error between y_true and y_pred
    """
    return bkd.mean(bkd.square(y_true - y_pred))


def mean_lp_error(y_pred, y_true, p=2):
    """
    calculate the mean Lp error between y_true and y_pred
    p is the order of the norm
    """
    return bkd.mean(bkd.norm(y_true - y_pred, p, dim=None))


def mean_lp_relative_error(y_pred, y_true, p=2):
    """
    calculate the mean Lp relative error between y_true and y_pred
    p is the order of the norm
    """
    return bkd.mean(bkd.norm(y_true - y_pred, p, dim=None) / bkd.norm(y_true, p, dim=None))


def softmax_cross_entropy(y_pred, y_true):
    """
    calculate the softmax cross entropy between y_true and y_pred
    """
    return bkd.softmax(y_true, dim=-1) * bkd.lgamma(bkd.softmax(y_pred, dim=-1))


LOSS_DICT = {
    "mean absolute error": mean_absolute_error,
    "MAE": mean_absolute_error,
    "mae": mean_absolute_error,
    "l1": mean_absolute_error,
    "L1": mean_absolute_error,
    "mean squared error": mean_squared_error,
    "MSE": mean_squared_error,
    "mse": mean_squared_error,
    "l2": mean_absolute_error,
    "L2": mean_absolute_error,
    "mean Lp error": mean_lp_error,
    "mean lp error": mean_lp_error,
    "lp": mean_lp_error,
    "Lp": mean_lp_error,
    "mean Lp relative error": mean_lp_relative_error,
    "mean lp relative error": mean_lp_relative_error,
    "lp relative": mean_lp_relative_error,
    "Lp relative": mean_lp_relative_error,
    "mean absolute percentage error": mean_absolute_percentage_error,
    "MAPE": mean_absolute_percentage_error,
    "mape": mean_absolute_percentage_error,
    "softmax cross entropy": softmax_cross_entropy,
    "softmax": softmax_cross_entropy,
}


def get(identifier):
    """Retrieves a loss function.

    Args:
        identifier: A loss identifier. String name of a loss function, or a loss function.

    Returns:
        A loss function.
    """
    if isinstance(identifier, (list, tuple)):
        return list(map(get, identifier))

    if isinstance(identifier, str):
        return LOSS_DICT[identifier]
    if callable(identifier):
        return identifier
    raise ValueError("Could not interpret loss function identifier:", identifier)


if __name__ == "__main__":

    x = bkd.ones((3, 4), dtype=bkd.float32)
    y = bkd.zeros((3, 4), dtype=bkd.float32)

    print(mean_absolute_error(x, y))
    print(mean_absolute_error(x, y))
    print(mean_lp_relative_error(x, y))
