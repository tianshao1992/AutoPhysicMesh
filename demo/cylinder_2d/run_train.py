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
from demo.cylinder_2d.config.default import get_config
from data_loader import get_dataloader
from net_module import NavierStokes2DSolver, NavierStokes2DEvaluator

all_config = get_config()
train_loaders, valid_loaders = get_dataloader(all_config)
netsolver = NavierStokes2DSolver(all_config)

board_config = all_config.Board
wandb.init(project=board_config.project, name=board_config.name, dir=board_config.dir)
evaluator = NavierStokes2DEvaluator(all_config, board=wandb)

netsolver.train(train_loaders, valid_loaders, evaluator)