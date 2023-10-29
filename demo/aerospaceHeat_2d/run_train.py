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
from Dataset.preprocess import DataNormer

from all_config import get_config
from data_loader import get_dataloader
from net_work import Heat2DPredictor, Heat2DEvaluator


all_config = get_config()
train_loaders, valid_loaders, test_loaders = get_dataloader(all_config)

netfitter = Heat2DPredictor(all_config)
evaluator = Heat2DEvaluator(all_config)

wandb_config = all_config.wandb
wandb.init(project=wandb_config.project, name=wandb_config.name, dir=wandb_config.dir)

netfitter.train(train_loaders, valid_loaders, input_normalizer, output_normalizer, evaluator)










