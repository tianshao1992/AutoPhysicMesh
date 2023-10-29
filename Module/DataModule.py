#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
# @Copyright (c) 2023 Baidu.com, Inc. All Rights Reserved
# @Time    : 2023/10/23 16:46
# @Author  : Liu Tianyuan (liutianyuan02@baidu.com)
# @File    : DataDriven.py
# @Description    : ******
"""

import os
import time
from abc import abstractmethod
from ml_collections import ConfigDict
from typing import Union, List, Dict, Tuple, Callable, Optional

from Module import bkd, nn, NNs
from Module._base import BasicModule, BaseEvaluator
from Module.NNs.optimizers import get as get_optimizer
from Module.NNs.lossfuncs import get as get_loss_func
from Module.NNs.weightning import initial_loss_weights, update_loss_weights, get_total_loss

Type_net_model = Union[dict, list, tuple, nn.ModuleDict, nn.ModuleList, nn.Module]

class NetFitter(BasicModule):
    def __init__(self,
                 config: ConfigDict,
                 models: Type_net_model,
                 params: ConfigDict = {},
                 *args,
                 **kwargs):
        r"""
            NetFitter
            Args:
                :param config:
                :param models:
                :param params:
                :param kwargs:
        """
        # the models should be a dict or list or tuple or nn.ModuleDict or nn.ModuleList or nn.Module
        if isinstance(models, nn.ModuleList or list or tuple):
            net_model = nn.ModuleDict()
            for model in models:
                net_model[model.__class__.__name__] = model
        else:
            net_model = models
        models = net_model.to(config.Device)
        super(NetFitter, self).__init__(config=config, models=models, params=params)
        self.config_setup(config)


    @abstractmethod
    def losses(self, batch, *args, **kwargs):
        # pass
        return NotImplementedError("Subclasses should implement this!")

    @abstractmethod
    def infer(self, batch):
        # pass
        return NotImplementedError("Subclasses should implement this!")

    @abstractmethod
    def valid(self, batch):
        # pass
        return NotImplementedError("Subclasses should implement this!")


    def train(self, train_loaders, valid_loaders, log_evaluator, *args, **kwargs):
        r"""
            train a BaseRegression instance.
            :param train_loaders:
            :param valid_loaders:
            :param evaluator:
            :param args:
            :param kwargs:
            :return:
        """
        time_sta = time.time()
        for epoch in range(1, self.config.Training.max_epoch+1):
            self.models.train()
            for batch in train_loaders:
                batch = train_loaders.batch_preprocess(batch)
                self.train_step(epoch, batch)

            if epoch % self.config.Logging.log_every_steps == 0:
                self.models.eval()
                # train_batch logger
                time_end = time.time()
                train_batch = next(iter(train_loaders))
                train_batch = train_loaders.batch_preprocess(train_batch)
                train_batch = self.valid(train_batch)
                train_batch = train_loaders.batch_postprocess(train_batch)
                self.update_states(prefix_name='train')
                log_evaluator.step(epoch, self.state_dict, train_batch, time_sta, time_end)

                # valid_batch logger
                time_sta = time.time()
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

    def train_step(self, epoch, batch, *args, **kwargs):
        r"""
            train a BaseModule in one step.
            :param epoch:
            :param batch:
            :param args:
            :param kwargs:
            :return:
        """
        self.optimizer.zero_grad()
        self.losses(batch)
        # Update weights if necessary
        if epoch % self.config.Training.Weighting.update_every_steps == 0:
            self.loss_weights, self.adapt_dict = (
                update_loss_weights(self.loss_weights,
                                    self.loss_dict,
                                    models=self.models,
                                    scheme=self.config.Training.Weighting.scheme,
                                    momentum=self.config.Training.Weighting.momentum))
        total_loss = get_total_loss(self.loss_dict, self.loss_weights)
        total_loss.backward()
        self.optimizer.step()
        self.total_loss = total_loss.item()

        return total_loss.item()


    def config_setup(self, config):

        self.models.to(config.Device)
        self.register_states()
        self.register_params()
        self.set_optimizer()
        self.set_loss_funcs()
        # self.callbacks = CallbackList(callbacks=callbacks)

    def save_model(self, path):
        save_path = path # os.path.join(os.path.split(os.path.abspath(path))[0], path)
        # if not os.path.exists(save_path):
        #     os.makedirs(save_path)
        bkd.save(
            {'config': self.config,
                  'models': self.models,
                  'loss_funcs': self.loss_funcs,
                  'optimizer': self.optimizer,
                  'scheduler': self.scheduler,
                  'params': self.params},
                   save_path)

    def load_model(self, path):
        checkpoint = bkd.load(path, map_location=self.config.Device)
        self.config = checkpoint['config']
        self.models = checkpoint['models']
        self.params = checkpoint['parmas']
        self.optimizer = checkpoint['optimizer']
        self.scheduler = checkpoint['scheduler']
        self.loss_funcs = checkpoint['loss_funcs']

    def set_optimizer(self, network_params=None):

        if network_params is None:
            network_params = self.models.parameters()

        # todo: support multi parameters for multi-optimizers and schedulers to different tasks
        if 'task_names' in self.config.Optim.keys():
            self.optimizer = {}
            self.scheduler = {}
            for task in self.config.Optim.task_names:
                if task not in self.config.Optim.optimizer.keys():
                   raise ValueError("the task name {} is not in the optimizer config!".format(task))
                else:
                    try:
                        self.optimizer[task] = get_optimizer(self.config.Optim.optimizer[task].name,
                                                              params=network_params,
                                                              **self.config.Optim.optimizer[task].params)
                        self.scheduler[task] = get_optimizer(self.config.Optim.scheduler[task].name,
                                                              optimizer=self.optimizer[task],
                                                              **self.config.Optim.scheduler[task].params)
                    except:
                        raise ValueError("the optimizers or scheduler config is not correct!")
        else:
            self.optimizer = get_optimizer(self.config.Optim.optimizer.name,
                                            params=network_params,
                                            **self.config.Optim.optimizer.params)
            self.scheduler = get_optimizer(self.config.Optim.scheduler.name,
                                            optimizer=self.optimizer,
                                            **self.config.Optim.scheduler.params)

    def set_loss_funcs(self):

        if 'task_names' in self.config.Loss.keys():
            self.loss_funcs = {}
            for task in self.config.Loss.task_names:
                if task not in self.config.Loss.keys():
                    raise ValueError("the task name {} is not in the loss config!".format(task))
                else:
                    try:
                        if 'params' in self.config.Loss[task].keys():
                            params = self.config.Loss[task].params
                        else:   # if the loss has no params
                            params = {}
                        self.loss_funcs[task] = get_loss_func(self.config.Loss[task].name, **params)
                    except:
                        raise ValueError("the loss config is not correct!")

        else:
            if 'params' in self.config.Loss.keys():
                params = self.config.Loss.params
            else:   # if the loss has no params
                params = {}
            self.loss_funcs = get_loss_func(self.config.Loss.name, **params)

        # todo: support multi loss weights for different tasks
        self.loss_dict = {}
        self.adapt_dict = {}
        if 'init_weights' in self.config.Training.Weighting.keys():
            self.loss_weights = self.config.Training.Weighting.init_weights
            self.total_loss = get_total_loss(self.loss_dict, self.loss_weights)
        else:
            self.loss_weights = None
            self.total_loss = None


    def update_states(self, prefix_name):
        """
        register_states for the model.
        """
        self.state_dict.prefix_name = prefix_name
        self.state_dict.time = time.time()
        self.state_dict.mode = self.config.Mode
        self.state_dict.device = self.config.Device
        self.state_dict.loss_dict = self.loss_dict
        self.state_dict.loss_weights = self.loss_weights
        self.state_dict.total_loss = self.total_loss
        self.state_dict.adapt_dict = self.adapt_dict
        self.state_dict.learning_rates = self.scheduler.get_last_lr()[0]


class NetFitterEvaluator(BaseEvaluator):
    def __init__(self,
                 config: ConfigDict,
                 board):

        super(NetFitterEvaluator, self).__init__(config, board)


    def log_losses(self, state, batch, *args, **kwargs):
        losses = state.loss_dict
        prefix = state.prefix_name
        for key, values in losses.items():
            self.log_dict[prefix + "_" + key + "_loss_values"] = values
        self.log_dict[prefix + "_" + "total_loss_values"] = state.total_loss

    def log_weights(self, state, batch, *args, **kwargs):
        loss_weights = state.loss_weights
        prefix = state.prefix_name
        if loss_weights is not None:
            for key, values in loss_weights.items():
                self.log_dict[prefix + "_" + key + "_loss_weights"] = values

    def log_lrs(self, state, batch, *args, **kwargs):
        lrs = state.learning_rates
        # for key, values in lrs.items():
        self.log_dict["global_learning_rates"] = lrs

    def log_adapt(self, state, batch, *args, **kwargs):
        adapt_dict = state.adapt_dict
        prefix = state.prefix_name
        for key, values in adapt_dict.items():
            self.log_dict[prefix + "_" + key + "_adapt_dict"] = values

    def step(self, epoch, state, batch, start_time, end_time, *args, **kwargs):
        self.log_dict = {}
        self.log_dict.update({
            "global_epoch": epoch,
            "cost_time": end_time - start_time,
            "mode": state.mode,
            "compute_device": state.device,
            "prefix_name": state.prefix_name,
        })

        if self.config.Logging.log_losses:
            self.log_losses(state, batch)
            self.log_lrs(state, batch)

        if self.config.Logging.log_metrics:
            self.log_metrics(state, batch)

        if self.config.Logging.log_weights:
            self.log_weights(state, batch)

        if self.config.Logging.log_plots:
            self.log_plots(state, batch)

        if self.config.Logging.log_params:
            self.log_params(state, batch)

        if self.config.Logging.log_adapt:
            self.log_adapt(state, batch)

        self.log_print(start_time, end_time, self.log_dict)
        self.board.log(self.log_dict, step=epoch)