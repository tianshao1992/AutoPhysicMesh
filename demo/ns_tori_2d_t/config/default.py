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
import math

def get_config():
    """Get the default hyperparameter configuration."""
    config = ml_collections.ConfigDict()
    config.Mode = "train"
    config.Seed = 2023
    config.Device = "cuda:0"

    # Weights & Biases
    config.Board = Board = ml_collections.ConfigDict()
    Board.project = "PINN-NS_tori_pytorch"
    Board.name = "standard"
    Board.dir = '../work'
    Board.tag = None

    # Normalization

    # Physics
    config.physics = physics = ml_collections.ConfigDict()
    physics.nu = 0.01  # vis
    # scale to [0, 1], update in dataloader
    physics.X = 2 * math.pi
    physics.Y = 2 * math.pi
    physics.T = 10.0

    # Arch
    config.Network = Network = ml_collections.ConfigDict()
    Network.arch_name = "ModifiedMlp"
    Network.input_dim = 3
    Network.layer_depth = 4
    Network.layer_width = 128
    Network.output_dim = 2
    Network.layer_active = "tanh"  # gelu works better than tanh for this problem
    Network.modified_mode = True
    Network.periodicity = None
    Network.periods_emb = ml_collections.ConfigDict({'input_dim': 3,
                                                     'output_dim': 16,
                                                     "scale": 1.0,
                                                     "axis": (0, 1)})

    Network.fourier_emb = ml_collections.ConfigDict({'input_dim': 16,
                                                     'output_dim': Network.layer_width,
                                                     'hidden_dim': 16,
                                                     "scale": 1.0,
                                                     "modes": 4})

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
    Training.max_epoch = 150000
    Training.ics_batch_size = 1
    Training.res_batch_size = 1
    Training.valid_batch_size = 1

    # Weighting
    config.Training.Weighting = Weighting = ml_collections.ConfigDict()
    Weighting.scheme = "gdn"
    Weighting.init_weights = {
        "u_ic": 1.0,
        "v_ic": 1.0,
        "w_ic": 1.0,
        "r_m": 1.0,
        "r_c": 1.0,
    }
    Weighting.momentum = 0.9
    Weighting.update_every_steps = 1000  # 100 for grad norm and 1000 for ntk
    Weighting.use_causal = False
    Weighting.causal_tol = 1.0
    Weighting.causal_num = 16

    # Logging
    config.Logging = Logging = ml_collections.ConfigDict()
    Logging.dir = Board.dir
    Logging.log_every_steps = 100
    Logging.log_losses = True
    Logging.log_metrics = True
    Logging.log_weights = True
    Logging.log_plots = True
    Logging.log_params = False
    Logging.log_adapt = False

    # keywords: Saving
    config.Saving = Saving = ml_collections.ConfigDict()
    Saving.save_every_steps = 10000
    Saving.save_path = Board.dir

    return config