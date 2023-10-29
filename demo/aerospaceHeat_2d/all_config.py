#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
# @Copyright (c) 2023 Baidu.com, Inc. All Rights Reserved
# @Time    : 2023/10/17 1:11
# @Author  : Liu Tianyuan (liutianyuan02@baidu.com)
# @File    : config.py
# @Description    : ******
"""

import ml_collections
import torch.nn


def get_config():
    """Get the default hyperparameter configuration."""

    config = ml_collections.ConfigDict()

    # the first of is the keywords which must set the this config
    config.Mode = "train"
    config.Seed = 2023
    config.Device = "cuda:0"

    # Weights & Biases
    config.Board = Board = ml_collections.ConfigDict()
    Board.project = "DataDriven_Heat2D"
    Board.name = "default"
    Board.dir = './work'
    Board.tag = None

    # Physics
    config.physics = physics = ml_collections.ConfigDict()
    physics.probe_star = 1
    physics.probe_step = 3
    physics.spatial_dim = 2

    # Arch define the network structure
    config.Network = Network = ml_collections.ConfigDict()

    # keywords: network
    config.Network.reconstruct = reconstruct = ml_collections.ConfigDict()
    reconstruct.arch_name = "FNO2d"
    reconstruct.input_dim = 1
    reconstruct.output_dim = 1
    reconstruct.spatial_dim = 2
    reconstruct.spectral_modes = 4
    reconstruct.layer_depth = 4
    reconstruct.layer_width = 64
    reconstruct.layer_active = "gelu"  # gelu works better than tanh for this problem

    config.Network.identify = identify = ml_collections.ConfigDict()
    identify.arch_name = "MLP"
    identify.input_dim = 90
    identify.layer_depth = 4
    identify.layer_width = 64
    identify.output_dim = 2
    identify.layer_active = "gelu"  # gelu works better than tanh for this problem
    identify.modified_mode = False

    # keywords: optim
    config.Optim = Optim = ml_collections.ConfigDict()

    # if task_names are not in the config, then the optimizer will be set to None

    Optim.optimizer = ml_collections.ConfigDict()
    Optim.optimizer.name = "Adam"
    Optim.optimizer.params = ml_collections.ConfigDict()
    Optim.optimizer.params.betas = (0.9, 0.999)
    Optim.optimizer.params.eps = 1e-8
    Optim.optimizer.params.lr = 1e-3
    Optim.optimizer.params.weight_decay = 0.0
    Optim.scheduler = ml_collections.ConfigDict()
    Optim.scheduler.name = "StepLR"
    Optim.scheduler.params = ml_collections.ConfigDict()
    Optim.scheduler.params.gamma = 0.9
    Optim.scheduler.params.step_size = 2000

    # keywords: Loss
    config.Loss = Loss = ml_collections.ConfigDict()
    Loss.task_names = ["reconstruct", "identify"]

    # task 1: reconstruct
    Loss.reconstruct = reconstruct = ml_collections.ConfigDict()
    reconstruct.name = "mse"
    # reconstruct.params = ml_collections.ConfigDict()
    # reconstruct.params.reduction = "mean"

    # task 2: identify
    Loss.identify = identify = ml_collections.ConfigDict()
    identify.name = "mse"
    # identify.params = ml_collections.ConfigDict()
    # identify.params.reduction = "mean"

    # keywords:  Training
    config.Training = Training = ml_collections.ConfigDict()
    Training.max_epoch = 300
    Training.train_size = 1000
    Training.valid_size = 100
    Training.train_batch_size = 256
    Training.valid_batch_size = 128
    # training loss weights
    config.Training.Weighting = Weighting = ml_collections.ConfigDict()
    Weighting.scheme = None
    Weighting.init_weights = None
    Weighting.momentum = 0.9
    Weighting.update_every_steps = 100  # 100 for grad norm and 1000 for ntk


    # keywords: Logging
    config.Logging = Logging = ml_collections.ConfigDict()
    Logging.dir = Board.dir
    Logging.log_every_steps = 5
    Logging.log_losses = True
    Logging.log_metrics = False
    Logging.log_weights = True
    Logging.log_plots = True
    Logging.log_params = False
    Logging.log_adapt = False

    # keywords: Saving
    config.Saving = Saving = ml_collections.ConfigDict()
    Saving.save_every_steps = 50
    Saving.save_path = Board.dir

    return config