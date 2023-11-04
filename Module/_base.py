#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
# @Copyright (c) 2023 Baidu.com, Inc. All Rights Reserved
# @Time    : 2023/07/25 10:40
# @Author  : Liu Tianyuan (liutianyuan02@baidu.com)
# @File    : _base.py
# @Description    : ******
"""

import os
from abc import abstractmethod
from typing import Union, List, Dict, Tuple, Callable, Optional
from ml_collections import ConfigDict
from Logger import Printer, Visual
from Metrics import MetricsManager
from Dataset.data.params import ParamsData
from Utilizes.commons import default

class BasicModule(object):
    """ Model BaseClass """
    def __init__(self,
                 config: ConfigDict,
                 models: object,
                 params: Tuple[ParamsData] = (),
                 *args,
                 **kwargs):

        self.config = config
        self.models = models
        self.register_params(params)
        self.register_states()

    def register_params(self, params):
        r"""
            register_params to generate self.params
        """
        self.params = ConfigDict()
        self.params.variable = ConfigDict()
        self.params.constant = ConfigDict()
        for param in params:
            for name in param.names:
                if getattr(param, name).mode == 'variable':
                    self.params.variable[name] = getattr(param, name)
                elif getattr(param, name).mode == 'constant':
                    self.params.constant[name] = getattr(param, name)

    def register_states(self):
        """
        set logger for the model.
        """
        self.state_dict = ConfigDict()

    @abstractmethod
    def load_model(self, model_file):
        """ load model """
        pass

    @abstractmethod
    def save_model(self, model_file):
        """ save model to file """
        pass

    @abstractmethod
    def train(self, data_loader):
        """
        train a BaseRegression instance.
        Any non-abstract model inherited from this class should implement this method.

        Args:
            data_loader(data_loader): Train set, including the input and label.
        """
        pass

    @abstractmethod
    def train_step(self, train_loaders):

        """
        train a BaseRegression instance.
        Any non-abstract model inherited from this class should implement this method.

        Args:
            train_loaders: Train set, including the input and label.
        """
        pass


    @abstractmethod
    def solve(self, data_loader):
        """
        train a BaseRegression instance.
        Any non-abstract model inherited from this class should implement this method.

        Args:
            data_loader(data_loader): Train set, including the input and label.
        """
        pass


    @abstractmethod
    def solve_step(self):
        r"""

        :return:
        """
        pass


    @abstractmethod
    def infer(self, data_loader):
        """
        inference a BaseRegression instance.

        Any non-abstract model inherited from this class should implement this method.

        Args:
            data_loader(data_loader): data_loader, including the input only.
        """
        pass


    def infer_step(self):
        """
        inference a BaseRegression instance.

        Any non-abstract model inherited from this class should implement this method.

        Args:
            data_loader(data_loader): data_loader, including the input only.
        """
        pass


    def update_states(self, states):
        """
        update logger for the model.
        """
        self.state_dict.update(states)

    def set_params(self):
        """
        set params for the model.
        """
        pass

    def get_params(self):
        """
        get params for the model.
        """
        pass

    def register_callbacks(self, callbacks):
        self.callbacks = callbacks


class BaseCallback:
    """Callback base class.

    Attributes:
        model: instance of ``Model``. Reference of the model being trained.
    """

    def __init__(self):
        self.model = None

    def set_model(self, model):
        if model is not self.model:
            self.model = model
            self.init()

    def init(self):
        """Init after setting a model."""

    def on_epoch_begin(self):
        """Called at the beginning of every epoch."""

    def on_epoch_end(self):
        """Called at the end of every epoch."""

    def on_batch_begin(self):
        """Called at the beginning of every batch."""

    def on_batch_end(self):
        """Called at the end of every batch."""

    def on_train_begin(self):
        """Called at the beginning of model training."""

    def on_train_end(self):
        """Called at the end of model training."""

    def on_infer_begin(self):
        """Called at the beginning of prediction."""

    def on_infer_end(self):
        """Called at the end of prediction."""

    def on_valid_begin(self):
        """Called at the beginning of validation."""

    def on_valid_end(self):
        """Called at the end of validation."""


class BaseCallbackList(BaseCallback):
    """Container abstracting a list of callbacks.

    Args:
        callbacks: List of ``Callback`` instances.
    """

    def __init__(self, callbacks=None):
        callbacks = callbacks or []
        self.callbacks = list(callbacks)
        self.model = None


    def on_epoch_begin(self):
        for callback in self.callbacks:
            callback.on_epoch_begin()

    def on_epoch_end(self):
        for callback in self.callbacks:
            callback.on_epoch_end()

    def on_batch_begin(self):
        for callback in self.callbacks:
            callback.on_batch_begin()

    def on_batch_end(self):
        for callback in self.callbacks:
            callback.on_batch_end()

    def on_train_begin(self):
        for callback in self.callbacks:
            callback.on_train_begin()

    def on_train_end(self):
        for callback in self.callbacks:
            callback.on_train_end()

    def on_infer_begin(self):
        for callback in self.callbacks:
            callback.on_infer_begin()

    def on_infer_end(self):
        for callback in self.callbacks:
            callback.on_infer_end()

    def on_valid_begin(self):
        for callback in self.callbacks:
            callback.on_valid_begin()

    def on_valid_end(self):
        for callback in self.callbacks:
            callback.on_valid_end()

    def append(self, callback):
        if not isinstance(callback, BaseCallback):
            raise Exception(str(callback) + " is an invalid Callback object")
        self.callbacks.append(callback)


class BaseEvaluator(object):
    """ Evaluator BaseClass """
    def __init__(self,
                 config: ConfigDict,
                 board,
                 *args,
                 **kwargs):

        self.config = config
        self.board = board
        self.visual = Visual(config)
        self.metric = MetricsManager(config.Metrics)
        self.printer = Printer(config)
        self.log_step = {}

        assert 'dir' in config.Logging.keys(), "dir must exist in logging configs!"
        if not os.path.exists(config.Logging.dir):
            os.makedirs(config.Logging.dir)

    @abstractmethod
    def log_info(self, *args, **kwargs):
        pass

    @abstractmethod
    def log_metrics(self, *args, **kwargs):
        pass

    @abstractmethod
    def log_losses(self, *args, **kwargs):
        pass


    def log_params(self, *args, **kwargs):
        pass


    def log_plots(self, *args, **kwargs):
        pass

    @abstractmethod
    def step(self, *args, **kwargs):
        pass

    @abstractmethod
    def log_print(self, start_time, end_time, print_dict):
        self.printer.print_info(start_time, end_time, print_dict)
