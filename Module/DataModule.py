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
# import warnings
from abc import abstractmethod
from ml_collections import ConfigDict
from typing import Union, List, Dict, Tuple, Callable, Optional

from Module import bkd, nn, NNs
from Module._base import BasicModule, BaseEvaluator
from Module.NNs.optimizers import get as get_optimizer
from Module.NNs.lossfuncs import get as get_loss_func
from Module.NNs.weightning import update_loss_weights, get_total_loss
from Metrics.MetricSystem import MetricsManager

Type_net_model = Union[dict, list, tuple, nn.ModuleDict, nn.ModuleList, nn.Module]

class NetFitter(BasicModule):
    def __init__(self,
                 config: ConfigDict,
                 models: Type_net_model,
                 params: tuple = (),
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
        r"""
            calculate training losses for the model.
            Args:
                :param batch:
            Return
                loss_dict
        """
        return NotImplementedError("Subclasses should implement this!")

    @abstractmethod
    def metrics(self, batch, *args, **kwargs):
        r"""
            calculate valid metrics for the model.
            Args:
                :param batch:
            Return
                metric_dict
        """
        # pass

        return NotImplementedError("Subclasses should implement this!")

    @abstractmethod
    def infer(self, batch):
        r"""
            infer the model.
            Args:
                :param batch:
            Return
                batch
        """
        # pass
        return NotImplementedError("Subclasses should implement this!")

    def valid(self, data_loaders, *args, **kwargs):
        self.metric_evals.clear()
        for batch in data_loaders:
            batch = data_loaders.batch_preprocess(batch)
            batch = self.valid_step(batch)
            batch = data_loaders.batch_postprocess(batch)
            batch_dict = self.metrics(batch)
            self.metric_dict = self.metric_evals.batch_update(batch_dict)
        return batch

    def valid_step(self, batch):
        batch = self.infer(batch)
        return batch

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
        for epoch in range(1, self.config.Training.max_epoch + 1):
            self.models.train()
            for batch in train_loaders:
                batch = train_loaders.batch_preprocess(batch)
                self.train_step(epoch, batch)

            if epoch % self.config.Logging.log_every_steps == 0:
                self.models.eval()
                # train_batch logger
                time_end = time.time()
                train_batch = self.valid(train_loaders)
                self.update_states(prefix_name='train')
                log_evaluator.step(epoch, self.state_dict, train_batch, time_sta, time_end)

                # valid_batch logger
                time_sta = time.time()
                valid_batch = self.valid(valid_loaders)
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
        batch_dict = self.losses(batch)
        # Update weights if necessary
        try:
            if epoch % self.config.Training.Weighting.update_every_steps == 0:
                self.loss_weights, self.adapt_dict = (
                    update_loss_weights(self.loss_weights,
                                        batch_dict,
                                        models=self.models,
                                        scheme=self.config.Training.Weighting.scheme,
                                        momentum=self.config.Training.Weighting.momentum))
        except:
            Warning("the loss weights are not updated!")

        total_loss = get_total_loss(batch_dict, self.loss_weights)
        total_loss.backward()
        self.optimizer.step()
        self.total_loss = total_loss.item()
        # todo: update loss_dict for each batch like metrics_dict
        self.loss_dict = batch_dict

        return total_loss.item()

    def config_setup(self, config):

        # self.models.to(config.Device)
        # self.register_states()
        # self.register_params(params)
        self.set_params()
        self.set_optimizer()
        self.set_loss_funcs()
        self.set_metric_evals()
        # self.callbacks = CallbackList(callbacks=callbacks)

    def save_model(self, path):
        save_path = path  # os.path.join(os.path.split(os.path.abspath(path))[0], path)
        # if not os.path.exists(save_path):
        #     os.makedirs(save_path)
        bkd.save(
            {'config': self.config,
             'models': self.models,
             'params': self.params,
             'loss_funcs': self.loss_funcs,
             'metric_evals': self.metric_evals,
             'optimizer': self.optimizer,
             'scheduler': self.scheduler,
             },
            save_path)

    def load_model(self, path):
        r"""
            load model
            Args:
                :param path: file path
        """
        checkpoint = bkd.load(path, map_location=self.config.Device)

        try:
            self.config = checkpoint['config']
        except:
            raise ValueError("the config is not correct!")
        try:
            self.models = checkpoint['models']
        except:
            Warning("the models are not loaded!")
        try:
            self.params = checkpoint['params']
        except:
            Warning("the params are not loaded!")

        try:
            self.optimizer = checkpoint['optimizer']
        except:
            Warning("the optimizer are not loaded!")
        try:
            self.scheduler = checkpoint['scheduler']
        except:
            Warning("the scheduler are not loaded!")

        try:
            self.loss_funcs = checkpoint['loss_funcs']
        except:
            Warning("the loss_funcs are not loaded!")

        try:
            self.metric_evals = checkpoint['metric_evals']
        except:
            Warning("the metrics are not loaded!")

    def set_optimizer(self, network_params=None):
        assert 'Optim' in self.config, "No Optim config found, the Optimizer are not defined in the config!"
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
            # whether to use the same optimizer and scheduler for all the tasks
            self.optimizer = get_optimizer(self.config.Optim.optimizer.name,
                                           params=network_params,
                                           **self.config.Optim.optimizer.params)
            self.scheduler = get_optimizer(self.config.Optim.scheduler.name,
                                           optimizer=self.optimizer,
                                           **self.config.Optim.scheduler.params)

    def set_loss_funcs(self):
        assert 'Loss' in self.config, "No Loss config found, the loss_functions are not defined in the config!"
        if 'task_names' in self.config.Loss.keys():
            self.loss_funcs = {}
            for task in self.config.Loss.task_names:
                if task not in self.config.Loss.keys():
                    raise ValueError("the task name {} is not in the loss config!".format(task))
                else:
                    try:
                        if 'params' in self.config.Loss[task].keys():
                            params = self.config.Loss[task].params
                        else:  # if the loss has no params
                            params = {}
                        self.loss_funcs[task] = get_loss_func(self.config.Loss[task].name, **params)
                    except:
                        raise ValueError("the loss config is not correct!")

        else:
            # whether to use the same optimizer and scheduler for all the tasks

            if 'params' in self.config.Loss.keys():
                params = self.config.Loss.params
            else:  # if the loss has no params
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

    def set_metric_evals(self):
        # set metrics for the model.
        assert 'Metrics' in self.config, "No Metrics config found, the Metrics will not be calculated!"
        self.metric_evals = MetricsManager(self.config.Metrics)
        self.metric_dict = {}

    def set_params(self):
        # set the value of the param to the models
        for key, param in self.params.constant.items():
            # todo: support multi types of params
            # note that parameter names are unique.
            self.models.register_buffer(name=key, tensor=bkd.tensor(param.data, dtype=bkd.float32))

        for key, param in self.params.variable.items():
            # todo: support multi types of params
            # if key in self.models.get_parameter_names():
            #     warnings.WarningMessage("the param {} is already in the models, ")
            # self.models.__setattr__(key, bkd.tensor(param.data, dtype=bkd.float32))
            # note that parameter names are unique.
            param_nn = nn.Parameter(bkd.tensor(param.data, dtype=bkd.float32))
            self.models.register_parameter(name=key, param=param_nn)

    def update_params(self):
        # get the value of the param from the model
        # note: the constant params are not updated
        for key, value in self.params.variable.items():
            self.params.variable[key].data = self.models.__getattr__(key).data.detach().cpu().numpy()
        return self.params

    def update_states(self, prefix_name: str = ''):
        """
        register_states for the model.
        """
        self.update_params()
        self.state_dict.params_variable = self.params.variable
        self.state_dict.params_constant = self.params.constant

        self.state_dict.prefix_name = prefix_name + '-' if prefix_name != '' else ''
        self.state_dict.time = time.time()
        self.state_dict.mode = self.config.Mode
        self.state_dict.device = self.config.Device
        self.state_dict.loss_dict = self.loss_dict
        self.state_dict.metric_dict = self.metric_dict
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
            self.log_dict[prefix + key + "_loss_values"] = values
        self.log_dict[prefix + "total_loss_values"] = state.total_loss

    def log_weights(self, state, batch, *args, **kwargs):
        loss_weights = state.loss_weights
        prefix = state.prefix_name
        if loss_weights is not None:
            for key, values in loss_weights.items():
                self.log_dict[prefix + key + "_loss_weights"] = values

    def log_lrs(self, state, batch, *args, **kwargs):
        lrs = state.learning_rates
        # for key, values in lrs.items():
        self.log_dict["global_learning_rates"] = lrs

    def log_adapt(self, state, batch, *args, **kwargs):
        adapt_dict = state.adapt_dict
        prefix = state.prefix_name
        for key, values in adapt_dict.items():
            self.log_dict[prefix + key + "_adapt_dict"] = values

    def log_metrics(self, state, batch, *args, **kwargs):
        metric_dict = state.metric_dict
        prefix = state.prefix_name
        for key, values in metric_dict.items():
            # note: to ensure the metric values to upload to the visual board is a scalar
            # todo: support metrics upload to the board as a vector to visual
            self.log_dict[prefix + key + "_metric_values"] = values.mean()

    def log_params(self, state, batch, *args, **kwargs):
        # todo: add necessary params to the log_dict, not all the params
        params_variable = state.params_variable
        params_constant = state.params_constant
        for key, values in params_variable.items():
            self.log_dict[key + "_params_values"] = values.data
        for key, values in params_constant.items():
            self.log_dict[key + "_params_values"] = values.data

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
