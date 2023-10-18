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
    wandb.project = "PINN-NS_unsteady_cylinder_pytorch"
    wandb.name = "default"
    wandb.tag = None

    # Nondimensionalization
    config.nondim = True

    # Physics
    config.physics = physics = ml_collections.ConfigDict()
    physics.L = 1.0
    physics.W = 1.0
    physics.T = 1.0
    physics.Re = 100.0

    # Arch
    config.network = network = ml_collections.ConfigDict()
    network.arch_name = "ModifiedMlp"
    network.input_dim = 3
    network.layer_depth = 4
    network.layer_width = 256
    network.output_dim = 3
    network.layer_active = "gelu"  # gelu works better than tanh for this problem
    network.modified_mode = True
    network.periodicity = None
    network.fourier_emb = ml_collections.ConfigDict({'input_dim': 3,
                                                     'output_dim': network.layer_width,
                                                     'hidden_dim': 8,
                                                     "scale": 1.0,
                                                     "modes": 2})

    # Optim
    config.optim = optim = ml_collections.ConfigDict()
    optim.optimizer = "Adam"
    optim.beta1 = 0.9
    optim.beta2 = 0.999
    optim.eps = 1e-8
    optim.learning_rate = 1e-3
    optim.decay_rate = 0.9
    optim.decay_steps = 2000
    optim.grad_accum_steps = 0

    # Training
    config.training = training = ml_collections.ConfigDict()
    training.max_steps = 200000
    training.num_time_windows = 10

    training.inflow_batch_size = 2048
    training.outflow_batch_size = 2048
    training.noslip_batch_size = 2048
    training.ic_batch_size = 2048
    training.res_batch_size = 4096

    # Weighting
    config.weighting = weighting = ml_collections.ConfigDict()
    weighting.scheme = "grad_norm"
    weighting.init_weights = {
        "u_ic": 1.0,
        "v_ic": 1.0,
        "p_ic": 1.0,
        "u_in": 1.0,
        "v_in": 1.0,
        "u_out": 1.0,
        "v_out": 1.0,
        "u_noslip": 1.0,
        "v_noslip": 1.0,
        "ru": 1.0,
        "rv": 1.0,
        "rc": 1.0,
    }

    weighting.momentum = 0.9
    weighting.update_every_steps = 1000  # 100 for grad norm and 1000 for ntk

    weighting.use_causal = True
    weighting.causal_tol = 1.0
    weighting.num_chunks = 16

    # Logging
    config.logging = logging = ml_collections.ConfigDict()
    logging.log_every_steps = 100
    logging.log_errors = True
    logging.log_losses = True
    logging.log_weights = True
    logging.log_grads = False
    logging.log_ntk = False
    logging.log_preds = False

    # Saving
    config.saving = saving = ml_collections.ConfigDict()
    saving.save_every_steps = 10000
    saving.num_keep_ckpts = 10

    # Input shape for initializing Flax models
    config.input_dim = 3

    # Integer for PRNG random seed.
    config.seed = 42

    return config