#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
# @Copyright (c) 2023 Baidu.com, Inc. All Rights Reserved
# @Time    : 2023/08/17 0:17
# @Author  : Liu Tianyuan (liutianyuan02@baidu.com)
# @File    : pinn.py
# @Description    : ******
"""

import os
import time
from abc import abstractmethod
from ml_collections import ConfigDict
from typing import Union, List, Dict, Tuple, Callable, Optional

from Module import bkd, nn, NNs
from Module.DataModule import NetFitter, NetFitterEvaluator

Type_net_model = Union[dict, list, tuple, nn.ModuleDict, nn.ModuleList, nn.Module]

class PinnSolver(NetFitter):
    def __init__(self,
                 config: ConfigDict,
                 models: Type_net_model,
                 params: ConfigDict = {},
                 **kwargs):

        super(PinnSolver, self).__init__(config=config, models=models, params=params)

    @abstractmethod
    def forward(self, inn_var, *args, **kwargs):
        # pass
        return NotImplementedError("Subclasses should implement this!")

    @abstractmethod
    def residual(self, batch, *args, **kwargs):
        # pass
        return NotImplementedError("Subclasses should implement this!")

    @abstractmethod
    def losses(self, batch, *args, **kwargs):
        # pass
        return NotImplementedError("Subclasses should implement this!")

    def train(self, train_loaders, valid_loaders, log_evaluator):

        time_sta = time.time()
        for epoch in range(1, self.config.Training.max_epoch+1):
            self.models.train()
            for batch in train_loaders:
                batch = train_loaders.batch_preprocess(batch)
                self.train_step(epoch, batch)

            if epoch % self.config.Logging.log_every_steps == 0:
                # no need for train batch validation in pinns
                # time_end = time.time()
                # self.update_states(prefix_name='train')
                # log_evaluator.step(epoch, self.state_dict, batch, time_sta, time_end)

                self.models.eval()
                valid_batch = next(iter(valid_loaders))
                valid_batch = valid_loaders.batch_preprocess(valid_batch)
                valid_batch = self.valid(valid_batch)
                valid_batch = valid_loaders.batch_postprocess(valid_batch)
                self.update_states(prefix_name='valid')
                time_end = time.time()
                log_evaluator.step(epoch, self.state_dict, valid_batch, time_sta, time_end)
                time_sta = time.time()

            # note: scheduler step should be after one epoch not one batch
            self.scheduler.step()

            # todo: support early stopping and best model saving
            if epoch % self.config.Saving.save_every_steps == 0:
                self.save_model(os.path.join(self.config.Saving.save_path, 'models.pth'))


class PinnEvaluator(NetFitterEvaluator):
    def __init__(self,
                 config: ConfigDict,
                 board):

        super(PinnEvaluator, self).__init__(config, board)