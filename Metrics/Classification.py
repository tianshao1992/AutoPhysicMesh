#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
# @Copyright (c) 2023 Baidu.com, Inc. All Rights Reserved
# @Time    : 2023/11/2 13:40
# @Author  : Liu Tianyuan (liutianyuan02@baidu.com)
# @File    : Classification.py
# @Description    : ******
"""

from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

from Metrics._base import BasicMetric

class ConfusionMatrix(BasicMetric):
    """Confusion Matrix.

    Args:
    Attributes:
        _NAME(str): Metric name.
        _MAXIMIZE(bool): Identify optimization direction.
    """
    _NAME = "ConfusionMatrix"

    def __init__(self,):
        super(ConfusionMatrix, self).__init__()

    def calculate(self, pred, true,
                  **kwargs):
        r"""
        Args:
            pred(np.ndarray): Predicted target values.
            true(np.ndarray): Ground truth (correct) target values
        """
        return confusion_matrix(true, pred)


class F1Score(BasicMetric):
    r"""
    Args:
    Attributes:
        _NAME(str): Metric name.
    """
    _NAME = "ConfusionMatrix"
    def __init__(self,):
        super(F1Score, self).__init__()

    def calculate(self, pred, true):
        r"""
        Args:
            pred(np.ndarray): Predicted target values.
            true(np.ndarray): Ground truth (correct) target values
        """
        return f1_score(true, pred)

class AUC(BasicMetric):
    r"""
    Args:
    Attributes:
        _NAME(str): Metric name.
    """

    _NAME = "AUC"
    def __init__(self,):
        super(AUC, self).__init__()

    def calculate(self, pred, true):
        r"""
        Args:
            pred(np.ndarray): Predicted target values.
            true(np.ndarray): Ground truth (correct) target values
        """
        return roc_auc_score(true, pred)