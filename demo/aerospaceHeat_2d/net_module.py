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
import wandb
import matplotlib.pyplot as plt

from Module import bkd, nn
from ModuleZoo.NNs.fno.FNOs import FNO
from ModuleZoo.NNs.mlp.MLPs import MlpNet
from Module.DataModule import Fitter, FitterEvaluator
from Module.NNs.lossfuncs import get as get_loss

from Utilizes.commons import fig2data


class Heat2DPredictor(DataFitter):
    def __init__(self, config):
        # super(NavierStokes2D, self).__init__()
        models = nn.ModuleDict()
        models['reconstruct'] = FNO(**config.network1)
        models['identify'] = FNO(**config.network2)
        super().__init__(config, models)

    def forward(self, inn_var):
        out_var = self.net_model(x=inn_var[..., 2:], grid=inn_var[..., :2])
        return out_var

    def losses(self, batch):

        reconstruct_batch = batch["reconstruct"]

        # reconstruct loss
        reconstruct_input = reconstruct_batch['input']
        reconstruct_true = reconstruct_batch['target']
        reconstruct_pred = self.forward(inn_var=reconstruct_input)
        reconstruct_loss = self.loss_func(reconstruct_pred, reconstruct_true)

        # identify loss
        identify_input = reconstruct_batch['input']
        identify_true = reconstruct_batch['target']
        identify_pred = self.forward(inn_var=reconstruct_input)
        identify_loss = self.loss_func(identify_pred, identify_true)

        self.loss_dict.update({
            "reconstruct_loss": reconstruct_loss,
            "identify_loss": identify_loss,
        })

        return self.loss_dict

    def infer(self, batch):

        temper_batch = batch["temper"]
        temper_input = temper_batch['input']

        self.net_model.eval()
        with bkd.no_grad():
            temper_pred = self.forward(inn_var=temper_input)

        return temper_pred

    def config_setup(self, config):

        self.loss_func = get_loss('mse')
        super().config_setup(config)


class Heat2DEvaluator(FitterEvaluator):
    def __init__(self, config, model):
        super(Heat2DEvaluator, self).__init__(config, model)

        pass

    def log_plot(self, batch, save_fig='pred_fields'):

        temper_batch = batch["temper"]
        temper_input = temper_batch['input']
        temper_true = temper_batch['true']
        temper_pred = temper_batch['pred']
        coords = temper_input[..., :2]

        for time_step in range(0, 60, 10):
            fig, axs = plt.subplots(3, 1, num=100+time_step, figsize=(10, 8))
            self.visual.plot_fields_2D(fig, axs, temper_true[time_step], temper_pred[time_step], coords[time_step],
                                       titles=['真实field', '预测field', '误差'], field_names=['T/℃'],
                                       cmaps=['jet', 'jet', 'coolwarm'])
            if isinstance(save_fig, str):
                fig.savefig(os.path.join(self.visual.save_path, save_fig + "_{}.jpg".format(time_step)))
            self.log_dict.update({'Predicted fields time step at {}'.format(time_step):
                                      wandb.Image(fig2data(fig))})
            plt.close(fig)
