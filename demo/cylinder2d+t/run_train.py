#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
# @Copyright (c) 2023 Baidu.com, Inc. All Rights Reserved
# @Time    : 2023/10/17 0:13
# @Author  : Liu Tianyuan (liutianyuan02@baidu.com)
# @File    : run_train.py
# @Description    : ******
"""

import time
import os
import wandb
import torch

from Module import bkd

from all_config import get_config
from data_loader import get_dataloader
from net_work import NavierStokes2DSolver, NavierStokes2DEvaluator

all_config = get_config()
train_loaders, valid_loaders = get_dataloader()
netsolver = NavierStokes2DSolver(all_config)
evaluator = NavierStokes2DEvaluator(all_config, netsolver)

wandb_config = all_config.wandb
wandb.init(project=wandb_config.project, name=wandb_config.name, dir=wandb_config.dir)

for epoch in range(all_config.training.max_epoch):

    time_sta = time.time()

    batch = {}
    for key, loader in train_loaders.items():
         sample_data = next(iter(loader))
         batch[key] = {'input': sample_data[0].to(all_config.network.device), 'target': sample_data[1].to(all_config.network.device)}

    netsolver.solve(epoch, batch)

    time_end = time.time()

    # for key, loader in valid_loaders.items():
    #     sample_data = next(iter(loader))
    #     batch[key] = {'input': sample_data[0].to(all_config.network.device), 'target': sample_data[1].to(all_config.network.device)}
    if epoch % all_config.logging.log_every_steps == 0:
        evaluator.step(epoch, time_sta, time_end, batch)
        wandb.log(evaluator.log_dict, step=epoch)

    if epoch % all_config.saving.save_every_steps == 0:
        netsolver.save_model("./save/net_model.pth")











