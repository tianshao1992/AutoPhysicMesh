#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
# @Copyright (c) 2023 Baidu.com, Inc. All Rights Reserved
# @Time    : 2023/10/17 0:17
# @Author  : Liu Tianyuan (liutianyuan02@baidu.com)
# @File    : pinn.py
# @Description    : ******
"""
import os
from abc import ABCMeta, abstractmethod
import ml_collections
from Module import bkd, nn
from ml_collections import ConfigDict

from Module import optimizers
from Logger.logger import Logger
from Logger.visualizer import Visual

from typing import Union, List, Dict, Tuple, Callable, Optional
net_model_type = Union[dict, list, tuple, nn.ModuleDict, nn.ModuleList, nn.Module]

class BasicSolver(object):
    def __init__(self, net_model: net_model_type,
                       config: ConfigDict,
                       **kwargs):
        super(BasicSolver, self).__init__()

        if isinstance(net_model, nn.Module):
            _net_model = {'net': net_model}
        elif isinstance(net_model, nn.ModuleList or list or tuple):
            _net_model = {}
            for model in net_model:
                _net_model[model.__class__.__name__] = model
        else:
            _net_model = net_model

        self.net_model = nn.ModuleDict(_net_model)
        self.config_setup(config)


    @abstractmethod
    def forward(self, *args, **kwargs):
        # pass
        return NotImplementedError("Subclasses should implement this!")
    @abstractmethod
    def residual(self, *args, **kwargs):
        # pass
        return NotImplementedError("Subclasses should implement this!")

    @abstractmethod
    def losses(self, batch):
        # pass
        return NotImplementedError("Subclasses should implement this!")

    @abstractmethod
    def infer(self, batch):
        # pass
        return NotImplementedError("Subclasses should implement this!")

    def solve(self, epoch, batch, *args, **kwargs):
        self.net_model.train()
        self.optimizer.zero_grad()
        self.losses(batch)
        # Update weights if necessary
        if self.config.weighting.scheme in ["gdn", "ntk"]:
            if epoch % self.config.weighting.update_every_steps == 0:
                self._update_loss_weights(momentum=self.config.weighting.momentum)
        total_loss = self._get_total_loss(batch, *args, **kwargs)
        self.total_loss = total_loss.item()
        total_loss.backward()
        self.optimizer.step()
        self.scheduler.step()

    def config_setup(self, config):
        self.config = config
        self.net_model.to(config.network.device)
        self._set_optimizer()
        if 'init_weights' in config.weighting.keys():
            self.loss_weights = config.weighting.init_weights
        else:
            self.loss_weights = None

    def save_model(self, path):
        save_path = os.path.split(os.path.abspath(path))[0]
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        bkd.save({'net_model': self.net_model,
                  'optimizer': self.optimizer,
                  'scheduler': self.scheduler, }, path)

    def load(self, path):
        checkpoint = bkd.load(path)
        self.net_model = checkpoint['net_model']
        self.optimizer = checkpoint['optimizer']
        self.scheduler = checkpoint['scheduler']
        self.net_model.to(self.config.network.device)
        self._set_optimizer()
        if 'init_weights' in self.config.weighting.keys():
            self.loss_weights = self.config.weighting.init_weights
        else:
            self.loss_weights = None

    def _get_total_loss(self, batch, *args, **kwargs):
        total_loss = 0.
        if self.loss_weights is None:
            self.loss_weights = {}
            for key, value in self.loss_dict.items():
                self.loss_weights[key] = 1.0
        for key, value in self.loss_dict.items():
            total_loss += value * self.loss_weights[key]
        self.total_loss = total_loss.item()
        return total_loss

    def _set_optimizer(self, network_params=None):
        if network_params is None:
            network_params = self.net_model.parameters()
        self.optimizer = optimizers.get(self.config.optim.optimizer.name,
                                        params=network_params,
                                        **self.config.optim.optimizer.params)
        self.scheduler = optimizers.get(self.config.optim.scheduler.name,
                                        optimizer=self.optimizer,
                                        **self.config.optim.scheduler.params)

    def _gdn_fn(self, loss):
        G = []
        grad_t = bkd.autograd.grad(loss, self.net_model.parameters(),
                                   retain_graph=True, create_graph=False)
        for item in grad_t:
            G.append(item.ravel().detach())
        return G

    def _ntk_fn(self, y):
        num = y.shape[-1] if len(y.shape) > 1 else 1
        K = []
        for i in range(num):
            grad = []
            grad_t = bkd.autograd.grad(y[..., (i,)], self.net_model.parameters(),
                                       retain_graph=True, create_graph=False)
            for item in grad_t:
                grad.extend(item.ravel())
            J = bkd.stack(grad)
            K.append(bkd.dot(J, J))
        return K

    def _update_loss_weights(self, momentum):
        if self.loss_weights is None:
            for key, value in self.loss_dict.items():
                self.loss_weights[key] = 1.0
        loss_weights = self._compute_loss_weights(self.config.weighting.scheme)

        for key, value in loss_weights.items():
            new_w = self.loss_weights[key] * momentum + (1 - momentum) * value.item()
            self.loss_weights.update({key: new_w})

        return self.loss_weights

    def _compute_loss_weights(self, scheme, pred_batch=None):

        if scheme == "gdn" or scheme == "grad_norm":
            gdn_list = []
            # Compute the gradient of each loss w.r.t. the parameters
            for key, value in self.loss_dict.items():
                # Compute the grad norm of each loss
                grads = self._gdn_fn(value)
                gdn_list.append(bkd.norm(bkd.cat(grads)))
            # Compute the mean of grad norms over all losses
            self._gdn_dict = dict(zip(self.loss_dict.keys(), gdn_list))
            mean_grad_norm = bkd.mean(bkd.stack(gdn_list))
            # Grad Norm Weighting
            w_value = list(map(lambda x: (mean_grad_norm / x), gdn_list))
            w = dict(zip(self.loss_dict.keys(), w_value))

        elif scheme == "ntk" or scheme == "ntk_norm":
            # Compute the diagonal of the NTK of each loss
            ntk_list = []
            for pred in pred_batch.values():
                ntk_list.extend(self._ntk_fn(pred))
            self._ntk_dict = dict(zip(self.loss_dict.keys(), ntk_list))
            # Compute the mean of the diagonal NTK corresponding to each loss
            mean_ntk_dict = map(lambda x: bkd.mean(x), ntk_list)
            # Compute the average over all ntk means
            mean_ntk = bkd.mean(bkd.cat(mean_ntk_dict))
            # NTK Weighting
            w_value = list(map(lambda x: (mean_ntk / x), mean_ntk_dict))
            w = dict(zip(self.loss_dict.keys(), w_value))

        return w

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)


class BaseEvaluator(object):
    def __init__(self, config: ConfigDict,
                       module: BasicSolver):
        self.config = config
        self.module = module
        self.visual = Visual(use_tex='ch-en')
        self.log_dict = {}
        self.logger = Logger()
        self.total_loss = 1e20

        if not os.path.exists(config.wandb.dir):
            os.mkdir(config.wandb.dir)

    def log_losses(self, batch, *args, **kwargs):
        losses = self.module.loss_dict
        for key, values in losses.items():
            self.log_dict[key + "_loss_values"] = values
        self.log_dict["total_loss_values"] = self.module.total_loss

    def log_weights(self, batch, *args, **kwargs):
        loss_weights = self.module.loss_weights
        for key, values in loss_weights.items():
            self.log_dict[key + "_loss_weights"] = values

    @abstractmethod
    def log_plot(self, batch, *args, **kwargs):
        pass

    def log_lrs(self, batch, *args, **kwargs):
        lrs = self.module.scheduler.get_last_lr()[0]
        # for key, values in lrs.items():
        self.log_dict["global_learning_rates"] = lrs

    def log_grads(self, batch, *args, **kwargs):
        gdn_dict = self.module._gdn_dict
        for key, values in gdn_dict.items():
            self.log_dict[key + "_loss_gdn"] = values

    def log_ntk(self, batch, *args, **kwargs):
        ntk_dict = self.module._ntk_dict
        for key, values in ntk_dict.items():
            self.log_dict[key + "_loss_ntk"] = values

    def step(self, epoch, time_sta, time_end, batch, *args, **kwargs):

        self.__call__(batch, *args, **kwargs)
        self.logger.log_print(epoch, time_sta, time_end, self.log_dict)

    def __call__(self, batch, *args, **kwargs):
        # Initialize the log dict
        self.log_dict = {}

        if self.config.logging.log_losses:
            self.log_losses(batch, *args, **kwargs)
            self.log_lrs(batch, *args, **kwargs)

        if self.config.logging.log_weights:
            self.log_weights(batch, *args, **kwargs)

        if self.config.logging.log_plot:
            self.log_plot(batch, *args, **kwargs)

        if self.config.logging.log_gdn:
            self.log_grads(batch, *args, **kwargs)

        if self.config.logging.log_ntk:
            self.log_ntk(batch, *args, **kwargs)

        return self.log_dict