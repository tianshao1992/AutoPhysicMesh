#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
# @Copyright (c) 2023 Baidu.com, Inc. All Rights Reserved
# @Time    : 2023/10/17 0:15
# @Author  : Liu Tianyuan (liutianyuan02@baidu.com)
# @File    : network.py
# @Description    : ******
"""

import os

import torch
import wandb
import matplotlib.pyplot as plt

from Module import bkd, nn
from ModuleZoo.NNs.fno.FNOs import FNO
from ModuleZoo.NNs.mlp.MLPs import MlpNet
from Module.DataModule import NetFitter, NetFitterEvaluator
from Module.NNs.lossfuncs import get as get_loss

from Utilizes.commons import fig2data


class Heat2DPredictor(NetFitter):
    def __init__(self, config):
        # super(NavierStokes2D, self).__init__()
        models = nn.ModuleDict()
        models['reconstruct'] = FNO(**config.Network.reconstruct)
        models['identify'] = MlpNet(**config.Network.identify)
        super(Heat2DPredictor, self).__init__(config, models)
        self.config_setup(config)

    def losses(self, batch):

        reconstruct_batch = batch["reconstruct"]
        # reconstruct loss
        reconstruct_input = reconstruct_batch['input_norm'].to(self.config.Device)
        reconstruct_true = reconstruct_batch['target_norm'].to(self.config.Device)
        reconstruct_pred = self.models['reconstruct'](x=reconstruct_input[..., 2:],
                                                      grid=reconstruct_input[..., :2])
        reconstruct_loss = self.loss_funcs['reconstruct'](reconstruct_pred, reconstruct_true)

        identify_batch = batch["identify"]
        # identify loss
        identify_input = identify_batch['input_norm'].to(self.config.Device)
        identify_true = identify_batch['target_norm'].to(self.config.Device)
        identify_pred = self.models['identify'](identify_input)
        identify_loss = self.loss_funcs['identify'](identify_pred, identify_true)

        self.loss_dict.update({
            "reconstruct_loss": reconstruct_loss,
            "identify_loss": identify_loss,
        })
        return self.loss_dict

    def valid(self, batch):
        with torch.no_grad():
            self.losses(batch)
            batch = self.infer(batch)
        return batch

    def infer(self, batch):
        with torch.no_grad():
            reconstruct_batch = batch["reconstruct"]
            # reconstruct loss
            reconstruct_input = reconstruct_batch['input_norm'].to(self.config.Device)
            reconstruct_pred = self.models['reconstruct'](x=reconstruct_input[..., 2:],
                                                          grid=reconstruct_input[..., :2])
            batch["reconstruct"].update({"pred": reconstruct_pred.cpu().numpy()})

            identify_batch = batch["identify"]
            # identify loss
            identify_input = identify_batch['input_norm'].to(self.config.Device)
            identify_pred = self.models['identify'](identify_input)
            batch["identify"].update({"pred": identify_pred.cpu().numpy()})
        return batch


class Heat2DEvaluator(NetFitterEvaluator):
    def __init__(self, config, board):
        super(Heat2DEvaluator, self).__init__(config, board)
        pass


    def log_plots(self, state, batch):

        prefix = state.prefix_name

        reconstruct_batch = batch["reconstruct"]
        reconstruct_input = reconstruct_batch['input']
        reconstruct_true = reconstruct_batch['target']
        reconstruct_pred = reconstruct_batch['pred']
        coords = reconstruct_input[..., :2]

        for time_step in range(0, 60, 10):
            fig, axs = plt.subplots(3, 1, num=100+time_step, figsize=(10, 8))
            self.visual.plot_fields_2D(fig, axs, reconstruct_true[time_step], reconstruct_pred[time_step],
                                       coords[time_step],
                                       titles=['真实field', '预测field', '误差'], field_names=['T/℃'],
                                       cmaps=['jet', 'jet', 'coolwarm'])


            fig.savefig(os.path.join(self.visual.save_path, prefix + "_fields_step_{}.jpg".format(time_step)))
            self.log_dict.update({prefix + ' predicted fields time step at {}'.format(time_step):
                                  wandb.Image(fig2data(fig))})
            plt.close(fig)


        identify_batch = batch["identify"]
        identify_input = identify_batch['input']
        identify_true = identify_batch['target']
        identify_pred = identify_batch['pred']

        fig, axs = plt.subplots(2, 2, num=200, figsize=(12, 12))

        for i in range(2):
            self.visual.plot_regression(fig, axs[0, i], identify_true[:, i], identify_pred[:, i],
                                        xylabels=('真实温度载荷/℃', '预测温度载荷/℃'))

            self.visual.plot_regression(fig, axs[1, i], identify_true[:, i], identify_pred[:, i],
                                        xylabels=('真实温度载荷/℃', '预测温度载荷/℃'))

        fig.savefig(os.path.join(self.visual.save_path,  prefix + "_identify.jpg"))
        self.log_dict.update({prefix + "_identify": wandb.Image(fig2data(fig))})
        plt.close(fig)