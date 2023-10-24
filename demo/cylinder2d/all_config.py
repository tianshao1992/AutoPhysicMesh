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
    wandb.project = "PINN-NS_steady_cylinder_pytorch"
    wandb.name = "default"
    wandb.dir = './work'
    wandb.tag = None

    # Normalization
    config.nondim = True
    # Physics
    config.physics = physics = ml_collections.ConfigDict()
    physics.nu = 0.001  # vis
    if not config.nondim:
        physics.L_star = 1.0  # characteristic velocity
        physics.U_star = 1.0  # characteristic length
    else:
        physics.L_star = 0.1  # characteristic velocity
        physics.U_star = 0.2  # characteristic length
    physics.Re = physics.U_star * physics.L_star / physics.nu  # Re
    # scale to [0, 1], update in dataloader
    physics.L = 1.0
    physics.W = 1.0
    physics.T = 1.0

    # Arch
    config.network = network = ml_collections.ConfigDict()
    network.arch_name = "ModifiedMlp"
    network.device = "cuda:0"
    network.input_dim = 2
    network.layer_depth = 4
    network.layer_width = 128
    network.output_dim = 3
    network.layer_active = "gelu"  # gelu works better than tanh for this problem
    network.modified_mode = True
    network.periodicity = None
    network.fourier_emb = ml_collections.ConfigDict({'input_dim': 2,
                                                     'output_dim': network.layer_width,
                                                     'hidden_dim': 8,
                                                     "scale": 1.0,
                                                     "modes": 2})

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
    training.max_epoch = 100000

    training.inflow_batch_size = 2048
    training.outflow_batch_size = 2048
    training.wall_batch_size = 2048
    training.cylinder_batch_size = 2048
    training.res_batch_size = 4096

    # Weighting
    config.weighting = weighting = ml_collections.ConfigDict()
    weighting.scheme = "gdn"
    weighting.init_weights = {
        "u_in": 1.0,
        "v_in": 1.0,
        "u_out": 1.0,
        "v_out": 1.0,
        "u_wall": 1.0,
        "v_wall": 1.0,
        "u_cylinder": 1.0,
        "v_cylinder": 1.0,
        "r_x": 1.0,
        "r_y": 1.0,
        "r_c": 1.0,
    }

    weighting.momentum = 0.9
    weighting.update_every_steps = 1000  # 100 for grad norm and 1000 for ntk

    # Logging
    config.logging = logging = ml_collections.ConfigDict()
    logging.log_every_steps = 100
    logging.log_errors = True
    logging.log_losses = True
    logging.log_weights = True
    logging.log_gdn = True
    logging.log_ntk = False
    logging.log_plot = True

    # Saving
    config.saving = saving = ml_collections.ConfigDict()
    saving.save_every_steps = 10000

    # Integer for PRNG random seed.
    config.seed = 42

    return config