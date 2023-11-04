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
from Business.htsy.CurvedPlate.config.default import get_config
from data_loader import get_dataloader
from net_module import Heat3DPredictor, Heat3DEvaluator

all_config = get_config()
train_loaders, valid_loaders, test_loaders = get_dataloader(all_config)

netfitter = Heat3DPredictor(all_config)

wandb_config = all_config.Board
wandb.init(project=wandb_config.project, name=wandb_config.name, dir=wandb_config.dir)
evaluator = Heat3DEvaluator(all_config, board=wandb)

netfitter.train(test_loaders, valid_loaders, evaluator)



