#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
# @Copyright (c) 2023 Baidu.com, Inc. All Rights Reserved
# @Time    : 2023/10/26 12:25
# @Author  : Liu Tianyuan (liutianyuan02@baidu.com)
# @File    : MetricModule.py
# @Description    : ******
"""

import numpy as np
from Metrics._base import BasicMetric


class Statistics(object):

    def __init__(self):
        self.clear()

    def clear(self):
        self.avg = 0
        self.sum = 0
        self.tol = 0
        self.avg = 0
        self.std = 0
        self.log = None

    def batch_update(self, val):
        assert isinstance(val, (float, int, complex, np.ndarray)), \
            "the input value should be a number or a numpy array!"
        if isinstance(val, (float, int, complex)):
            val = np.array([val, ])
        if self.log is None:
            self.log = val
        else:
            self.log = np.concatenate((self.log, val), axis=0)
        self.tol = len(self.log)
        self.sum = np.sum(self.log, axis=0)
        self.avg = self.sum / self.tol
        self.std = np.std(self.log, axis=0)


class MetricsManager(object):

    def __init__(self,
                 config,
                 *args,
                 **kwargs):

        r"""
        Args:
            config(Config): Config object.
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.
        """

        self.config = config

        self.metric_funcs = {}
        self.metric_stats = {}
        # todo: whether to log the sample data
        self.sample_dicts = {}


        if 'task_names' not in self.config:
            try:
                metric_config = self.config
                func_name = metric_config.name if 'name' in metric_config else None
                func_params = metric_config.params if 'params' in metric_config else {}
                self.register_metric('metric', func_name, **func_params)
            except:
                Warning("No Metrics config found, the Metrics will not be calculated!")
        else:
            for metric_name in self.config.task_names:
                if metric_name not in self.config:
                    raise ValueError("Metric {} is not in the Metrics config!".format(metric_name))
                metric_config = self.config[metric_name]
                func_name = metric_config.name if 'name' in metric_config else None
                func_params = metric_config.params if 'params' in metric_config else {}
                self.register_metric(metric_name, func_name, **func_params)

    def register_metric(self, metric_name, metric_func, **func_kwargs):
        r"""
            register a metric function
            Args:
                metric_name(str): name of the metric
                metric_func(str): name of the metric function
                func_kwargs(dict): keyword arguments of the metric function
        """
        if isinstance(metric_func, BasicMetric):
            func = metric_func
        else:
            func = BasicMetric.get_metrics_by_name(metric_func)
        self.metric_funcs.update({metric_name: func(**func_kwargs)})

    def clear(self):
        for metric_name in self.metric_stats.keys():
            self.metric_stats[metric_name].clear()

    def batch_update(self, batch_dicts):
        r"""
            update the metric statistics
            Args:
                batch_dicts(dict): dict of the metric values
        """
        metric_dicts = {}

        for name, value in batch_dicts.items():
            # todo: whether to log the sample data
            # todo: n = batch_size
            if name not in self.metric_stats:
                self.metric_stats.update({name: Statistics()})
            self.metric_stats[name].batch_update(value)
            # todo: return avg sum tol std?
            metric_dicts.update({name: self.metric_stats[name].avg})
        return metric_dicts

if __name__ == "__main__":
    from Regression import MSE
    metric = BasicMetric.get_metrics_by_name("MSE")()

    true = np.array([1, 2, 3, 4, 5])
    pred = np.array([1, 2, 3, 4, 5])

    r = metric.calculate(pred, true)


    metric = BasicMetric.get_metrics_by_name("MAE")()

    true = np.array([1, 2, 3, 4, 5])
    pred = np.array([1, 2, 3, 4, 5])

    r = metric.calculate(pred, true)


    metric = BasicMetric.get_metrics_by_name("MAPE")()

    true = np.array([1, 2, 3, 4, 5])
    pred = np.array([1, 2, 3, 4, 5])

    r = metric.calculate(pred, true)

    metric = BasicMetric.get_metrics_by_name("MSLE")()

    true = np.array([1, 2, 3, 4, 5])
    pred = np.array([1, 2, 3, 4, 5])

    r = metric.calculate(pred, true)

    metric = (BasicMetric.get_metrics_by_name("PhysicsLpMetric")
              (p=2, relative=False, channel_reduction=False, samples_reduction=False))

    true = np.ones((10, 10, 10, 1), dtype=np.float32)
    pred = np.ones((10, 10, 10, 1), dtype=np.float32)

    r = metric.calculate(pred, true)