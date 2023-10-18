#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
# @Copyright (c) 2023 Baidu.com, Inc. All Rights Reserved
# @Time    : 2023/10/17 0:17
# @Author  : Liu Tianyuan (liutianyuan02@baidu.com)
# @File    : pinn.py
# @Description    : ******
"""

from abc import ABCMeta, abstractmethod
from model import bkd, nn
from ml_collections import ConfigDict

from model import autograd

class BasicModule(object):
    def __init__(self, net_model: dict or list or tuple or nn.ModuleDict or nn.ModuleList or nn.Module,
                       config: ConfigDict,
                       **kwargs):
        super(BasicModule, self).__init__()

        if isinstance(net_model, nn.Module):
            _net_model = {'net': net_model}
        elif isinstance(net_model, nn.ModuleList or list or tuple):
            _net_model = {}
            for model in net_model:
                _net_model[model.__class__.__name__] = model
        else:
            _net_model = net_model

        self.net_model = nn.ModuleDict(_net_model)
        self.config = config
        self.loss_dict = {}
        self.ntk_dict = {}
        self.loss_weights = None


    @abstractmethod
    def forward(self, *args, **kwargs):
        # pass
        return NotImplementedError("Subclasses should implement this!")
    @abstractmethod
    def residual(self, *args, **kwargs):
        # pass
        return NotImplementedError("Subclasses should implement this!")

    @abstractmethod
    def losses(self, *args, **kwargs):
        # pass
        return NotImplementedError("Subclasses should implement this!")


    def ntk_fn(self, y):
        num = y.shape[-1] if len(y.shape) > 1 else 1
        K = []
        for i in range(num):
            grad = []
            grad_t = bkd.autograd.grad(y[..., (i,)], self.net_model.parameters(), retain_graph=True)
            for item in grad_t:
                grad.extend(item.ravel())
            J = bkd.stack(grad)
            K.append(bkd.dot(J, J))
        return K

    def _update_loss_weights(self, momentum):
        if self.loss_weights is None:
            self.loss_weights = [1.] * len(self.loss_dict)
        loss_weights = self._compute_loss_weights()
        running_average = (
            lambda old_w, new_w: old_w * momentum + (1 - momentum) * new_w
        )
        self.loss_weights = list(map(running_average, self.loss_weights, loss_weights))
        return self.loss_weights

    def _compute_loss_weights(self, scheme, pred_batch=None):
        grad_norm_list = []
        if scheme == "grad_norm":
            # Compute the gradient of each loss w.r.t. the parameters
            for key, value in self.loss_dict.items():
                # Compute the grad norm of each loss
                grad = []
                grad_t = autograd.jacobian(value, self.net_model.parameters(), retain_graph=True, create_graph=True)
                for item in grad_t:
                    grad.extend(item.ravel())
                grad_norm_list.append(bkd.norm(bkd.stack(grad)))
            # Compute the mean of grad norms over all losses
            mean_grad_norm = bkd.mean(bkd.stack(grad_norm_list))
            # Grad Norm Weighting
            w = list(map(lambda x: (mean_grad_norm / x), grad_norm_list))

        elif scheme == "ntk":
            # Compute the diagonal of the NTK of each loss
            ntk_list = []
            for pred in pred_batch.values():
                ntk_list.extend(self.ntk_fn(pred))
            self.ntk_dict = dict(zip(self.loss_dict.keys(), ntk_list))
            # Compute the mean of the diagonal NTK corresponding to each loss
            mean_ntk_dict = map(lambda x: bkd.mean(x), ntk_list)
            # Compute the average over all ntk means
            mean_ntk = bkd.mean(bkd.stack(mean_ntk_dict))
            # NTK Weighting
            w = list(map(lambda x: (mean_ntk / x), mean_ntk_dict))

        return w

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)