#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
# @Copyright (c) 2023 Baidu.com, Inc. All Rights Reserved
# @Time    : 2023/10/26 10:55
# @Author  : Liu Tianyuan (liutianyuan02@baidu.com)
# @File    : utils.py
# @Description    : ******
"""

import warnings
from Module import bkd, nn
from Utilizes.commons import default

DICT = ['gdn', 'ntk', 'grad_norm', 'ntk_norm']

def gdn_fn(loss, models):
    r"""
    Compute the gradient of each loss w.r.t. the parameters
        :param loss: loss tensor
        :param models: model parameters
        :return:
    """
    G = []
    grad_t = bkd.autograd.grad(loss, models.parameters(), retain_graph=True, create_graph=False)
    for item in grad_t:
        G.append(item.ravel().detach())
    return G

def ntk_fn(y, models):
    num = y.shape[-1] if len(y.shape) > 1 else 1
    K = []
    for i in range(num):
        grad = []
        grad_t = bkd.autograd.grad(y[..., (i,)], models.parameters(), retain_graph=True, create_graph=False)
        for item in grad_t:
            grad.extend(item.ravel())
        J = bkd.stack(grad)
        K.append(bkd.dot(J, J))
    return K


def get_total_loss(loss_dict, loss_weights):
    loss_weights = default(initial_loss_weights(loss_dict), loss_weights)
    total_loss = 0.0
    # todo: modify like jax tree_map
    for key, value in loss_dict.items():
        total_loss += value * loss_weights[key]
    return total_loss

def initial_loss_weights(loss_dict):
    loss_weights = {}
    for key, value in loss_dict.items():
        loss_weights[key] = 1.0
    return loss_weights

def update_loss_weights(loss_weights, loss_dict, models, scheme,
                        pred_batch=None, momentum=0.9):

    if loss_weights is None:
        loss_weights = {}
        for key, value in loss_dict.items():
            loss_weights[key] = 1.0
    if scheme in DICT:
        new_weights, log_dict = adapt_loss_weights(loss_dict, models, scheme, pred_batch)
        for key, value in new_weights.items():
            new_w = loss_weights[key] * momentum + (1 - momentum) * value.item()
            loss_weights.update({key: new_w})
        return loss_weights, log_dict
    else:
        warnings.warn("this adaptive scheme is not supported, the loss_weights is not updated!")
        return loss_weights, None

def adapt_loss_weights(loss_dict, models, scheme, pred_batch=None):

    assert isinstance(loss_dict, dict), "loss_dict must be a python dict!"
    if scheme == "gdn" or scheme == "grad_norm":
        gdn_list = []
        # Compute the gradient of each loss w.r.t. the parameters
        for key, value in loss_dict.items():
            # Compute the grad norm of each loss
            grads = gdn_fn(value, models)
            gdn_list.append(bkd.norm(bkd.cat(grads)))
        # Compute the mean of grad norms over all losses
        gdn_dict = dict(zip(loss_dict.keys(), gdn_list))
        mean_grad_norm = bkd.mean(bkd.stack(gdn_list))
        # Grad Norm Weighting
        w_value = list(map(lambda x: (mean_grad_norm / x), gdn_list))
        w = dict(zip(loss_dict.keys(), w_value))
        return w, gdn_dict

    elif scheme == "ntk" or scheme == "ntk_norm":
        # Compute the diagonal of the NTK of each loss
        ntk_list = []
        for pred in pred_batch.values():
            ntk_list.extend(ntk_fn(pred, models))
        ntk_dict = dict(zip(loss_dict.keys(), ntk_list))
        # Compute the mean of the diagonal NTK corresponding to each loss
        mean_ntk_dict = map(lambda x: bkd.mean(x), ntk_list)
        # Compute the average over all ntk means
        mean_ntk = bkd.mean(bkd.cat(mean_ntk_dict))
        # NTK Weighting
        w_value = list(map(lambda x: (mean_ntk / x), mean_ntk_dict))
        w = dict(zip(loss_dict.keys(), w_value))
        return w, ntk_dict
    else:
        raise NotImplementedError("this adaptive scheme is not supported!")