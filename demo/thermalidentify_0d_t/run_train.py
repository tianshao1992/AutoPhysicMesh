#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
# @Copyright (c) 2023 Baidu.com, Inc. All Rights Reserved
# @Time    : 2023/10/17 0:13
# @Author  : Liu Tianyuan (liutianyuan02@baidu.com)
# @File    : run_train.py
# @Description    : ******
"""
import wandb
from demo.thermalidentify_0d_t.conifg.default import get_config
from data_loader import get_dataloader
from net_module import HeatLaplace1DSolver, HeatLaplace1DEvaluator

all_config = get_config()
train_loaders, valid_loaders, physics_params = get_dataloader(all_config)
netsolver = HeatLaplace1DSolver(all_config, physics_params)

board_config = all_config.Board
wandb.init(project=board_config.project, name=board_config.name, dir=board_config.dir)
evaluator = HeatLaplace1DEvaluator(all_config, board=wandb)

netsolver.train(train_loaders, valid_loaders, evaluator)