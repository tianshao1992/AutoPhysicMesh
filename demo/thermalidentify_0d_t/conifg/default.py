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

    # Physics
    config.physics = physics = ml_collections.ConfigDict()
    physics.material_name = "P-2"
    physics.identify_mode = "equal"
    physics.heating_power = "small"
    physics.time_resolution = 3

    # Weights & Biases
    config.Board = Board = ml_collections.ConfigDict()
    Board.project = "formula-thermal-identify-pytorch"
    Board.name = physics.material_name
    Board.dir = './work'
    Board.tag = None

    # Arch
    config.Network = Network = ml_collections.ConfigDict()
    Network.arch_name = "analytical_formula"

    # Optim
    config.Optim = Optim = ml_collections.ConfigDict()
    Optim.optimizer = ml_collections.ConfigDict()
    Optim.optimizer.name = "Adam"
    Optim.optimizer.params = ml_collections.ConfigDict()
    Optim.optimizer.params.betas = (0.8, 0.9)
    Optim.optimizer.params.eps = 1e-8
    Optim.optimizer.params.lr = 1e-5
    Optim.optimizer.params.weight_decay = 0.0
    Optim.scheduler = ml_collections.ConfigDict()
    Optim.scheduler.name = "StepLR"
    Optim.scheduler.params = ml_collections.ConfigDict()
    Optim.scheduler.params.gamma = 0.9
    Optim.scheduler.params.step_size = 200

    # keywords: Loss
    config.Loss = Loss = ml_collections.ConfigDict()
    Loss.name = "mse"

    # keywords: Metrics
    config.Metrics = Metrics = ml_collections.ConfigDict()
    Metrics.task_names = ["reconstruct", 'identify']

    # task 1: reconstruct
    Metrics.reconstruct = reconstruct = ml_collections.ConfigDict()
    reconstruct.name = "PhysicsLpMetric"
    reconstruct.params = ml_collections.ConfigDict()
    reconstruct.params.p = 2
    reconstruct.params.relative = True
    reconstruct.params.channel_reduction = False
    reconstruct.params.samples_reduction = True

    # task 2: identify
    Metrics.identify = identify = ml_collections.ConfigDict()
    identify.name = "mape"

    # keywords: Training
    config.Training = Training = ml_collections.ConfigDict()
    Training.max_epoch = 400
    Training.train_batch_size = 300
    # Weighting
    config.Training.Weighting = Weighting = ml_collections.ConfigDict()
    Weighting.scheme = None
    Weighting.update_every_steps = 100  # 100 for grad norm and 1000 for ntk
    Weighting.momentum = 0.9


    # Logging
    config.Logging = Logging = ml_collections.ConfigDict()
    Logging.dir = Board.dir
    Logging.log_every_steps = 10
    Logging.log_losses = True
    Logging.log_metrics = True
    Logging.log_weights = True
    Logging.log_plots = True
    Logging.log_params = True
    Logging.log_adapt = False

    # keywords: Saving
    config.Saving = Saving = ml_collections.ConfigDict()
    Saving.save_every_steps = 50
    Saving.save_path = Board.dir

    return config