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

    # Weights & Biases
    config.wandb = wandb = ml_collections.ConfigDict()
    wandb.project = "DataDriven_Heat2D"
    wandb.name = "default"
    wandb.dir = './work'
    wandb.tag = None

    # Normalization
    config.nondim = True
    # Physics
    config.physics = physics = ml_collections.ConfigDict()
    physics.L = 1.0
    physics.W = 1.0
    physics.T = 1.0
    physics.Re = 100.0

    # Arch
    config.network = network = ml_collections.ConfigDict()
    network.arch_name = "FNO2d"
    network.device = "cuda:0"
    network.input_dim = 2
    network.output_dim = 1
    network.spatial_dim = 2
    network.spectral_modes = 4
    network.layer_depth = 4
    network.layer_width = 64
    network.layer_active = "gelu"  # gelu works better than tanh for this problem
    network.input_norm = True
    network.output_norm = True

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
    training.max_epoch = 200000

    # Weighting
    config.weighting = weighting = ml_collections.ConfigDict()
    weighting.scheme = "none"
    weighting.init_weights = None

    weighting.momentum = 0.9
    weighting.update_every_steps = 1000  # 100 for grad norm and 1000 for ntk

    # Logging
    config.logging = logging = ml_collections.ConfigDict()
    logging.log_every_steps = 100
    logging.log_errors = True
    logging.log_losses = True
    logging.log_weights = True
    logging.log_plot = True
    logging.log_gdn = False
    logging.log_ntk = False

    # Saving
    config.saving = saving = ml_collections.ConfigDict()
    saving.save_every_steps = 10000
    saving.num_keep_ckpts = 10

    # Integer for PRNG random seed.
    config.seed = 42

    return config