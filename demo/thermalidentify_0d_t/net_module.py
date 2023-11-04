#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
# @Copyright (c) 2023 Baidu.com, Inc. All Rights Reserved
# @Time    : 2023/10/30 12:56
# @Author  : Liu Tianyuan (liutianyuan02@baidu.com)
# @File    : net_module.py
# @Description    : ******
"""
import os
import matplotlib.pyplot as plt
import numpy as np
from Module import bkd, nn
from Module.DataModule import NetFitter, NetFitterEvaluator
from Utilizes.commons import fig2data

class Analytical_Formula(nn.Module):
    def __init__(self):
        super(Analytical_Formula, self).__init__()

    def forward(self, t):
        x = self.q0 * self.delta / self.lambdas * bkd.ones_like(t)
        pi = bkd.pi
        for i in range(30):
            x -= 8*self.q0 * self.delta / (pi ** 2) / self.lambdas *\
                 bkd.exp(-((2*i+1)/2*pi)**2 * self.lambdas / self.rho * self.cp * t / self.delta **2) / (2*i+1)**2
        return x

class HeatLaplace1DSolver(NetFitter):
    def __init__(self, config, params):
        # super(NavierStokes2D, self).__init__()

        net_model = Analytical_Formula()
        super(HeatLaplace1DSolver, self).__init__(config, models=net_model, params=params)
        # self.config_setup(config)

    def losses(self, batch):

        exp_batch = batch["exp"]
        # data loss
        exp_input = exp_batch['input'].to(self.config.Device)
        exp_true = exp_batch['target'].to(self.config.Device)
        exp_pred = self.models(exp_input)
        exp_loss = self.loss_funcs(exp_pred, exp_true)

        self.loss_dict.update({
            "exp": exp_loss,
        })

        return self.loss_dict

    def infer(self, batch):

        valid_batch = batch["exp"]
        res_input = valid_batch['input'].to(self.config.Device)
        res_pred = self.models(res_input)
        batch["exp"].update({"pred": res_pred.detach().cpu()})

        return batch

    def metrics(self, batch, *args, **kwargs):

        metric_dict = {}

        self.update_params()

        lambdas_true = np.array(self.params.constant['lambda0'].data)
        cp_true = np.array(self.params.constant['cp0'].data)

        lambdas_pred = np.array(self.params.variable['lambdas'].data)
        cp_pred = np.array(self.params.variable['cp'].data)

        exp_batch = batch["exp"]
        exp_true = exp_batch['target'][None]
        exp_pred = exp_batch['pred'][None]

        metric_dict.update({
            "temperature": self.metric_evals.metric_funcs['reconstruct'](exp_pred, exp_true),
            "lambdas": self.metric_evals.metric_funcs['identify'](lambdas_pred, lambdas_true),
            "cp": self.metric_evals.metric_funcs['identify'](1/cp_pred, 1/cp_true),
        })

        return metric_dict

class HeatLaplace1DEvaluator(NetFitterEvaluator):
    def __init__(self, config, board):
        super().__init__(config, board)
        pass

    def log_plots(self, state, batch, save_fig='pred_fields'):

        real = batch['exp']['target']
        pred = batch['exp']['pred']
        coords = batch['exp']['input']
        lmm = batch['LMM']['target']

        index = coords[..., 0].sort()[1]
        real = real[index]
        pred = pred[index]
        lmm = lmm[index]
        coords = coords[index]

        # draw the predicted fields
        fig, axs = plt.subplots(2, 1, num=100, figsize=(8, 8))
        self.visual.plot_fields_1D(fig, axs, real, pred, coords, xylabels=['t/s', 'T/K'],
                                   titles=['exp', '误差'])
        if isinstance(save_fig, str):
            fig.savefig(os.path.join(self.visual.save_path, "exp_whole.jpg"))
        plt.close(fig)
        self.log_dict.update({'compared with exp': self.board.Image(fig2data(fig))})

        # draw the pde residual loss
        fig, axs = plt.subplots(2, 1, num=100, figsize=(8, 8))
        self.visual.plot_fields_1D(fig, axs, lmm, pred, coords, xylabels=['t/s', 'T/K'],
                                   titles=['lmm', '误差'])
        if isinstance(save_fig, str):
            fig.savefig(os.path.join(self.visual.save_path, "lmm_whole.jpg"))
        plt.close(fig)
        self.log_dict.update({'compared with llm': self.board.Image(fig2data(fig))})

        # draw the predicted fields
        n_local = 100
        fig, axs = plt.subplots(2, 1, num=100, figsize=(8, 8))
        self.visual.plot_fields_1D(fig, axs, real[:n_local], pred[:n_local], coords[:n_local], xylabels=['t/s', 'T/K'],
                                   titles=['exp', '误差'])
        if isinstance(save_fig, str):
            fig.savefig(os.path.join(self.visual.save_path, "exp_local.jpg"))
        plt.close(fig)
        self.log_dict.update({'compared with exp at first 30 step': self.board.Image(fig2data(fig))})

        # draw the pde residual loss
        fig, axs = plt.subplots(2, 1, num=100, figsize=(8, 8))
        self.visual.plot_fields_1D(fig, axs, lmm[:n_local], pred[:n_local], coords[:n_local], xylabels=['t/s', 'T/K'],
                                   titles=['lmm', '误差'])
        if isinstance(save_fig, str):
            fig.savefig(os.path.join(self.visual.save_path, "lmm_local.jpg"))
        plt.close(fig)
        self.log_dict.update({'compared with llm at first 30 step': self.board.Image(fig2data(fig))})


