#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
# @Copyright (c) 2023 Baidu.com, Inc. All Rights Reserved
# @Time    : 2023/10/18 16:28
# @Author  : Liu Tianyuan (liutianyuan02@baidu.com)
# @File    : optimizers.py
# @Description    : ******
"""
__all__ = ['get']


from Model import bkd, nn
from torch.optim import Adam, SGD, RMSprop, Adagrad, Adadelta, AdamW, Adamax, ASGD, LBFGS, Rprop
from torch.optim.lr_scheduler import StepLR, MultiStepLR, ExponentialLR, CosineAnnealingLR, ReduceLROnPlateau

DICT = {'Adam': Adam, 'SGD': SGD, 'RMSprop': RMSprop, 'Adagrad': Adagrad,
        'Adadelta': Adadelta, 'AdamW': AdamW, 'Adamax': Adamax,
        'ASGD': ASGD, 'LBFGS': LBFGS, 'Rprop': Rprop,
        "StepLR": StepLR, "MultiStepLR": MultiStepLR,
        "ExponentialLR": ExponentialLR, "CosineAnnealingLR": CosineAnnealingLR,
        "ReduceLROnPlateau": ReduceLROnPlateau}


def get(identifier, *args, **kwargs):
    """Returns function.

    Args:
        identifier: Function or string.

    Returns:
        Function corresponding to the input string or input function.
    """

    if identifier is None:
        return Adam(*args, **kwargs)
    if isinstance(identifier, str):
        return DICT[identifier](*args, **kwargs)
    if callable(identifier):
        return identifier(*args, **kwargs)
    raise TypeError(
        "Could not interpret optimizer identifier: {}".format(identifier)
    )


if __name__ == "__main__":

    network = nn.Linear(10, 10)
    optimizer = get('Adam', params=network.parameters(), lr=0.1)
    scheduler = get('StepLR', optimizer=optimizer, step_size=10, gamma=0.1)

    from ml_collections import ConfigDict
    config = ConfigDict()

    config.optim = optim = ConfigDict()
    optim.optimizer = ConfigDict()
    optim.optimizer.name = "Adam"
    optim.optimizer.params = ConfigDict()
    optim.optimizer.params.betas = (0.9, 0.999)
    optim.optimizer.params.eps = 1e-8
    optim.optimizer.params.lr = 1e-3
    optim.optimizer.params.weight_decay = 0.0
    optim.scheduler = ConfigDict()
    optim.scheduler.name = "StepLR"
    optim.scheduler.params = ConfigDict()
    optim.scheduler.params.gamma = 0.9
    optim.scheduler.params.step_size = 2000

    optimizer = get(config.optim.optimizer.name, params=network.parameters(), **config.optim.optimizer.params)
    scheduler = get(config.optim.scheduler.name, optimizer=optimizer, **config.optim.scheduler.params)