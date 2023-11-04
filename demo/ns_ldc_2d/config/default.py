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


    # Physics
    config.physics = physics = ml_collections.ConfigDict()
    physics.Re = 3200 # [100, 400, 1000, 3200]

    # Weights & Biases
    config.Board = Board = ml_collections.ConfigDict()
    Board.project = "PINN-ldc_pytorch"
    Board.name = "Re_{}".format(physics.Re)
    Board.dir = './work'
    Board.tag = None

    # Arch
    config.Network = Network = ml_collections.ConfigDict()
    Network.arch_name = "ModifiedMlp"
    Network.input_dim = 2
    Network.layer_depth = 5
    Network.layer_width = 128
    Network.output_dim = 3
    Network.layer_active = "gelu"  # gelu works better than tanh for this problem
    Network.modified_mode = True
    Network.periods_emb = None
    Network.fourier_emb = ml_collections.ConfigDict({'input_dim': 2,
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

    # keywords: Metrics
    config.Metrics = Metrics = ml_collections.ConfigDict()
    Metrics.task_names = ["fields", "residual"]

    # task 1: fields
    Metrics.fields = fields = ml_collections.ConfigDict()
    fields.name = "PhysicsLpMetric"
    fields.params = ml_collections.ConfigDict()
    fields.params.p = 2
    fields.params.relative = True
    fields.params.channel_reduction = True
    fields.params.samples_reduction = True

    # task 2: residual
    Metrics.residual = residual = ml_collections.ConfigDict()
    residual.name = "PhysicsLpMetric"
    residual.params = ml_collections.ConfigDict()
    residual.params.p = 2
    residual.params.relative = False
    residual.params.channel_reduction = True
    residual.params.samples_reduction = True

    # keywords: Training
    config.Training = Training = ml_collections.ConfigDict()
    Training.max_epoch = 50000
    Training.bcs_batch_size = 4
    Training.res_batch_size = 64
    Training.valid_batch_size = 1

    # Weighting
    config.Training.Weighting = Weighting = ml_collections.ConfigDict()
    Weighting.scheme = "gdn"
    Weighting.init_weights = {
        "u_bc": 1.0,
        "v_bc": 1.0,
        "r_x": 1.0,
        "r_y": 1.0,
        "r_c": 1.0,
    }
    Weighting.momentum = 0.9
    Weighting.update_every_steps = 1000  # 100 for grad norm and 1000 for ntk

    # Logging
    config.Logging = Logging = ml_collections.ConfigDict()
    Logging.dir = Board.dir
    Logging.log_every_steps = 500
    Logging.log_losses = True
    Logging.log_metrics = True
    Logging.log_weights = True
    Logging.log_plots = True
    Logging.log_params = False
    Logging.log_adapt = False

    # keywords: Saving
    config.Saving = Saving = ml_collections.ConfigDict()
    Saving.save_every_steps = 5000
    Saving.save_path = Board.dir

    return config