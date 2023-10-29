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

def get_config():
    """Get the default hyperparameter configuration."""
    config = ml_collections.ConfigDict()
    config.mode = "train"
    config.seed = 2023

    # Weights & Biases
    config.wandb = wandb = ml_collections.ConfigDict()
    wandb.project = "DataDriven_Heat2D"
    wandb.name = "default"
    wandb.dir = './work'
    wandb.tag = None

    # Physics
    config.physics = physics = ml_collections.ConfigDict()
    physics.probe_star = 0
    physics.probe_step = 3
    physics.spatial_dim = 2

    # Arch
    config.network1 = network1 = ml_collections.ConfigDict()
    network1.arch_name = "FNO2d"
    network1.device = "cuda:0"
    network1.input_dim = 2
    network1.output_dim = 1
    network1.spatial_dim = 2
    network1.spectral_modes = 4
    network1.layer_depth = 4
    network1.layer_width = 64
    network1.layer_active = "gelu"  # gelu works better than tanh for this problem

    config.network2 = network2 = ml_collections.ConfigDict()
    network2.arch_name = "MLP"
    network2.input_dim = 2
    network2.layer_depth = 4
    network2.layer_width = 64
    network2.output_dim = 3
    network2.layer_active = "gelu"  # gelu works better than tanh for this problem
    network2.modified_mode = False

    # Optim
    config.optim = optim = ml_collections.ConfigDict()
    optim.optimizer = ml_collections.ConfigDict()
    optim.optimizer.name = "Adam"
    optim.optimizer.params = ml_collections.ConfigDict()
    optim.optimizer.params.betas = (0.9, 0.999)
    optim.optimizer.params.eps = 1e-8
    optim.optimizer.params.lr = 1e-3
    optim.optimizer.params.weight_decay = 0.0
    optim.scheduler = ml_collections.ConfigDict()
    optim.scheduler.name = "StepLR"
    optim.scheduler.params = ml_collections.ConfigDict()
    optim.scheduler.params.gamma = 0.9
    optim.scheduler.params.step_size = 2000

    # Training
    config.training = training = ml_collections.ConfigDict()
    training.max_epoch = 1000
    training.train_size = 1000
    training.valid_size = 100

    # Weighting
    config.weighting = weighting = ml_collections.ConfigDict()
    weighting.scheme = None
    weighting.init_weights = None

    weighting.momentum = 0.9
    weighting.update_every_steps = 1000  # 100 for grad norm and 1000 for ntk

    # Logging
    config.logging = logging = ml_collections.ConfigDict()
    logging.log_every_steps = 100
    logging.log_errors = True
    logging.log_losses = True
    logging.log_weights = True
    logging.log_plots = True
    logging.log_params = False
    logging.log_gdn = False
    logging.log_ntk = False

    # Saving
    config.saving = saving = ml_collections.ConfigDict()
    saving.save_every_steps = 10000

    return config