#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
# @Copyright (c) 2023 Baidu.com, Inc. All Rights Reserved
# @Time    : 2023/10/23 12:58
# @Author  : Liu Tianyuan (liutianyuan02@baidu.com)
# @File    : _base.py
# @Description    : ******
"""

import abc
class BasicModule(abc.ABC):

    """Module base class."""

    def config_setup(self, config):
        """
        setup the config for the model.
        Any non-abstract model inherited from this class should implement this method.

        Args:
            config: Config for the model
        """
        self.config = config

    def save_model(self, file_path: str):
        """
        save a BasicModule to a file.
        Any non-abstract model inherited from this class should implement this method.

        Args:
            file_path(str): The path of the file to save to
        Returns:
            None
        """
        pass

    def load_model(self, file_path: str):
        """
        load a BasicModule to a file.
        Any non-abstract model inherited from this class should implement this method.

        Args:
            file_path(str): The path of the file to save to
        Returns:
            None
        """
        pass

    def train(self, data_loader):
        """
        train a BaseRegression instance.
        Any non-abstract model inherited from this class should implement this method.

        Args:
            data_loader(data_loader): Train set, including the input and label.
        """
        pass

    def infer(self, data_loader):
        """
        inference a BaseRegression instance.

        Any non-abstract model inherited from this class should implement this method.

        Args:
            data_loader(data_loader): data_loader, including the input only.
        """
        pass

    def train_step(self, train_loader):

        """
        train a BaseRegression instance.
        Any non-abstract model inherited from this class should implement this method.

        Args:
            train_loader(BaseDataset): Train set, including the input and label.
        """
        pass

    def valid_step(self, valid_loader):
        """
        valid a BaseRegression instance.
        Any non-abstract model inherited from this class should implement this method.

        Args:
            valid_loader(Dataset|None): Eval set, used for early stopping.
        """
        pass


    def _set_lossfunc(self, lossfunc):
        """
        set loss function for the model.
        Args:
            lossfunc: Loss function in the training process
        """
        self.lossfunc = lossfunc


    def _set_optimizer(self, optimizer):
        """
        set optimizer for the model.

        Args:
            optimizer: Optimizer in the training process
        """
        self.optimizer = optimizer

    def _set_scheduler(self, scheduler):
        """
        set scheduler for the model.

        Args:
            scheduler: scheduler in the training process
        """
        self.scheduler = scheduler