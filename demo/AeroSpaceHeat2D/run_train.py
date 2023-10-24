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
import wandb

from Dataset.dataprocess import DataNormer

from all_config import get_config
from data_loader import get_dataloader
from net_work import Heat2DPredictor, Heat2DEvaluator


all_config = get_config()
train_loaders, valid_loaders = get_dataloader()

input_normalizer = DataNormer(train_loaders.datasets['temper'].input_data, method="min-max")
output_normalizer = DataNormer(train_loaders.datasets['temper'].output_data, method="min-max")

netsolver = Heat2DPredictor(all_config)
evaluator = Heat2DEvaluator(all_config, netsolver)

wandb_config = all_config.wandb
wandb.init(project=wandb_config.project, name=wandb_config.name, dir=wandb_config.dir)

time_sta = time.time()
for epoch in range(all_config.training.max_epoch):

    # for key, loader in train_loaders.items():
    #      sample_data = next(iter(loader))
    #      input_data = input_normalizer.norm(sample_data[0]).reshape(-1, 9, 10, 4)
    #      output_data = output_normalizer.norm(sample_data[1]).reshape(-1, 9, 10, 1)
    #
    #      train_batch[key] = {'input': input_data.to(all_config.network.device),
    #                          'target': output_data.to(all_config.network.device)}
    #
    train_batch = next(iter(train_loaders))
    input_data = input_normalizer.norm(sample_data[0]).reshape(-1, 9, 10, 4)
    output_data = output_normalizer.norm(sample_data[1]).reshape(-1, 9, 10, 1)

    train_batch['temp'] = {'input': input_data.to(all_config.network.device),
                           'target': output_data.to(all_config.network.device)}
    netsolver.solve(epoch, train_batch)


    if epoch % all_config.logging.log_every_steps == 0:

        valid_batch = {}
        for key, loader in valid_loaders.items():
            sample_data = next(iter(loader))
            input_data = input_normalizer.norm(sample_data[0]).reshape(-1, 9, 10, 4)
            output_data = output_normalizer.norm(sample_data[1]).reshape(-1, 9, 10, 1)

            valid_batch[key] = {'input': input_data.to(all_config.network.device),
                                'target': output_data.to(all_config.network.device)}

            output_pred = netsolver.infer(valid_batch)

            input_data = input_normalizer.back(input_data).cpu().numpy()
            output_true = output_normalizer.back(output_data).cpu().numpy()
            output_pred = output_normalizer.back(output_pred).cpu().numpy()

            valid_batch[key] = {'input': input_data,
                                'true': output_true,
                                'pred': output_pred}

            time_end = time.time()
            evaluator.step(epoch, time_sta, time_end, valid_batch)
            wandb.log(evaluator.log_dict, step=epoch)
            time_sta = time.time()

    if epoch % all_config.saving.save_every_steps == 0:
        netsolver.save_model("./net_model.pth")











