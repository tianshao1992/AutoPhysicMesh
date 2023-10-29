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
    config.Mode = "train"
    config.Seed = 2023
    config.Device = "cuda:0"

    # Weights & Biases
    config.Board = Board = ml_collections.ConfigDict()
    Board.project = "PINN-NS_steady_cylinder_pytorch"
    Board.name = "standard"
    Board.dir = './work'
    Board.tag = None

    # Normalization

    # Physics
    config.physics = physics = ml_collections.ConfigDict()
    physics.nondim = True
    physics.nu = 0.001  # vis
    if not physics.nondim:
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
    config.Network = Network = ml_collections.ConfigDict()
    Network.arch_name = "ModifiedMlp"
    Network.input_dim = 2
    Network.layer_depth = 4
    Network.layer_width = 128
    Network.output_dim = 3
    Network.layer_active = "gelu"  # gelu works better than tanh for this problem
    Network.modified_mode = True
    Network.periodicity = None
    Network.fourier_emb = ml_collections.ConfigDict({'input_dim': 2,
                                                     'output_dim': Network.layer_width,
                                                     'hidden_dim': 8,
                                                     "scale": 1.0,
                                                     "modes": 2})

    # Optim
    config.Optim = Optim = ml_collections.ConfigDict()
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
    Loss.name = "mse"

    # keywords: Training
    config.Training = Training = ml_collections.ConfigDict()
    Training.max_epoch = 50000
    Training.inflow_batch_size = 2048
    Training.outflow_batch_size = 2048
    Training.wall_batch_size = 2048
    Training.cylinder_batch_size = 2048
    Training.res_batch_size = 8192
    Training.valid_batch_size = 20000

    # Weighting
    config.Training.Weighting = Weighting = ml_collections.ConfigDict()
    Weighting.scheme = "gdn"
    Weighting.init_weights = {
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
    Weighting.momentum = 0.9
    Weighting.update_every_steps = 1000  # 100 for grad norm and 1000 for ntk

    # Logging
    config.Logging = Logging = ml_collections.ConfigDict()
    Logging.dir = Board.dir
    Logging.log_every_steps = 100
    Logging.log_losses = True
    Logging.log_metrics = False
    Logging.log_weights = True
    Logging.log_plots = True
    Logging.log_params = False
    Logging.log_adapt = False

    # keywords: Saving
    config.Saving = Saving = ml_collections.ConfigDict()
    Saving.save_every_steps = 10000
    Saving.save_path = Board.dir

    return config