#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
# @Copyright (c) 2023 Baidu.com, Inc. All Rights Reserved
# @Time    : 2023/11/2 11:06
# @Author  : Liu Tianyuan (liutianyuan02@baidu.com)
# @File    : commons.py
# @Description    : ******
"""
# !/usr/bin/env python3
# -*- coding: UTF-8 -*-


import numpy as np
import sklearn.metrics as metrics

from Metrics._base import BasicMetric

class PhysicsLpMetric(BasicMetric):
    """Mean Squared Error.

    Args:
        p: Lp-norm type.
        relative: Whether to use relative error.
        channel_reduction: Whether to reduce the channel dimension.
        samples_reduction: Whether to reduce the samples dimension.
        eps: A small value to avoid divide by zero.
    Attributes:
        _NAME(str): Metric name.
        _MAXIMIZE(bool): Identify optimization direction.
    """
    _NAME = "PhysicsLpMetric"

    def __init__(self,
                 p: int or str,
                 relative: bool,
                 channel_reduction: bool,
                 samples_reduction: bool,
                 eps=1e-10):
        super(PhysicsLpMetric, self).__init__()

        # Lp-norm type are positive
        assert p > 0, 'Lp-norm type should be positive!'

        self.ord = p
        self.eps = eps
        self.relative = relative
        self.channel_reduction = channel_reduction
        self.samples_reduction = samples_reduction

    def calculate(self, pred, true,
                  samples_weights=None,
                  channel_weights=None,
                  distribution_weights=None,
                  **kwargs):
        r"""
        Args:
            pred(np.ndarray): Predicted target values.
            true(np.ndarray): Ground truth (correct) target values.
            samples_weights(np.ndarray or float): Sample weights.
            channel_weights(np.ndarray or float): Channel weights.
            distribution_weights(np.ndarray or float): Distribution weights.
        """
        assert len(pred.shape) >= 2, 'The shape of pred and true should be at least 2D!'
        assert pred.shape == true.shape, 'The shape of pred and true should be the same!'

        n_samples = pred.shape[0]
        n_channel = pred.shape[-1]

        if samples_weights is None:
            samples_weights = 1.0
        if not isinstance(samples_weights, (int, float, complex)):
            assert n_samples == len(samples_weights), \
                'The shape[0] of true should be the same as samples_weights!'

        if channel_weights is None:
            channel_weights = 1.0
        if not isinstance(channel_weights, (int, float, complex)):
            assert n_samples == len(channel_weights), \
                'The shape[-1] of true should be the same as channel_weights!'

        err = pred.reshape(n_samples, -1, n_channel) - true.reshape(n_samples, -1, n_channel)
        n_distribution = err.shape[1]

        if distribution_weights is None:
            distribution_weights = 1.0
        if not isinstance(distribution_weights, (int, float, complex)):
            assert n_distribution == len(distribution_weights), \
                'The spatial or temporal dim of true should be the same as distribution_weights!'

        err_norms = np.linalg.norm(err * distribution_weights, self.ord, axis=1)

        if self.relative:
            all_norms = np.linalg.norm(true.reshape(n_samples, -1, n_channel), self.ord, axis=1)
            # avoid divide by zero
            eps = np.finfo(np.float64).eps
            # todo: support self.eps
            res_norms = err_norms / (all_norms + eps)
        else:
            res_norms = err_norms

        if self.samples_reduction:
            # note: the sample dims will be reduced to 1 to ensure the shape of res_norms is at least 1D.
            res_norms = np.mean(res_norms * samples_weights, axis=0, keepdims=True)

        if self.channel_reduction:
            res_norms = np.mean(res_norms * channel_weights, axis=-1)

        return res_norms


class MAE(BasicMetric):
    """Mean Absolute Error.

    Args:

    Attributes:
        _NAME(str): Metric name.
    """
    _NAME = "mae"

    def __init__(self):
        super(MAE, self).__init__()

    def calculate(self, pred, true, **kwargs) -> float:
        # todo: add more options for multi-input of the sklearn function
        return metrics.mean_absolute_error(true, pred)


class MSE(BasicMetric):
    """Mean Absolute Error.

    Args:

    Attributes:
        _NAME(str): Metric name.
    """
    _NAME = "mse"

    def __init__(self):
        super(MSE, self).__init__()

    def calculate(self, pred, true, **kwargs) -> float:
        # todo: add more options for multi-input of the sklearn function
        return metrics.mean_squared_error(true, pred)


class MAPE(BasicMetric):
    """Mean Absolute Percentage Error.

    Args:

    Attributes:
        _NAME(str): Metric name.
    """
    _NAME = "mape"

    def __init__(self):
        super(MAPE, self).__init__()

    def calculate(self, pred, true, **kwargs) -> np.ndarray or float:
        # todo: add more options for multi-input of the sklearn function
        return metrics.mean_absolute_percentage_error(true, pred)


class MSLE(BasicMetric):
    r"""
    Mean Squared Log Error.
    """
    _NAME = "msle"

    def __init__(self):
        super(MSLE, self).__init__(ord=2,
                                   relative=False,
                                   channel_reduction=True,
                                   samples_reduction=True)

    def calculate(self, pred, true, **kwargs):
        r"""
        Args:
            pred(np.ndarray): Predicted target values.
            true(np.ndarray): Ground truth (correct) target values.
        """
        return metrics.mean_squared_log_error(true, pred)

if __name__ == "__main__":
    # from Regression import MSE
    metric = MSE.get_metrics_by_name("MSE")()