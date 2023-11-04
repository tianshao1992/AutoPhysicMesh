#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
# @Copyright (c) 2023 Baidu.com, Inc. All Rights Reserved
# @Time    : 2023/11/2 10:39
# @Author  : Liu Tianyuan (liutianyuan02@baidu.com)
# @File    : _base.py
# @Description    : ******
"""


from typing import Any, List, Tuple, Dict
from abc import ABC, abstractmethod
import numpy as np
from Module.utils import tensor2numpy


class BasicMetric(ABC):
    """Abstract base class used to build new Metric.

        Args:
            mode(str): Supported metric modes, only normal, prob and anomaly are valid values.
                       to support more modes, please override the forward method.
            kwargs: Keyword parameters of specific metric functions.
        """

    def __init__(self, **kwargs):
        pass

    @abstractmethod
    def calculate(self, pred, true, **kwargs) -> np.ndarray or float:
        """
        Compute metric's value from ndarray. todo: add support for tensor?
        Args:
            true(np,ndarray): Estimated target values.
            pred(np.ndarray): Ground truth (correct) target values.
        Returns:
            np.ndarray: Metric value.
        Raises:
            ValueError.
        """
        return NotImplementedError("Base class forward method is not implemented")

    def __call__(self, pred, true):
        r"""
            Args:
                y_true(np.ndarray): Ground truth (correct) target values.
                y_pred(np,ndarray): Estimated target values.
        """

        _pred = tensor2numpy(pred)
        _true = tensor2numpy(true)

        if isinstance(_true, (int, float, complex)):
            _true = np.ones_like(pred) * _true

        if _pred.shape == ():  # _pred is a Singleton array
            _pred = np.array([_pred, ])

        if _true.shape == ():  # _true is a Singleton array
            _true = np.array([_true, ])

        assert _pred.shape == _true.shape, "pred and true must have the same shape!"
        metric = self.calculate(_pred, _true)
        return metric


    @classmethod
    def get_metrics_by_name(cls, name: str) -> 'BasicMetric':
        """Get list of metric classes.

        Args:
            names(List[str]): List of metric names.

        Returns:
            List[Metric]: List of metric classes.
        """
        available_metrics = cls.__subclasses__()
        available_names = [metric._NAME for metric in available_metrics]
        assert (name in available_names
                ), f"{name} is not available, choose in {available_names}"
        idx = available_names.index(name)
        metric = available_metrics[idx]
        return metric

