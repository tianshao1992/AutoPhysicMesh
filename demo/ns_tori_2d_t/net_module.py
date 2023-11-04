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
from ModuleZoo.NNs.mlp.MLPs import FourierEmbedding, PeriodsEmbedding, MlpNet
from Module.FusionModule import PinnSolver, PinnEvaluator
from Module.NNs.autograd import gradient
from Module.NNs.weightning import casual_loss_weights
from Utilizes.commons import fig2data

class NavierStokes2DSolver(PinnSolver):
    def __init__(self, config, ):
        # super(NavierStokes2D, self).__init__()

        transforms = []
        if config.Network.periods_emb is not None:
            transforms.append(PeriodsEmbedding(**config.Network.periods_emb))

        if config.Network.fourier_emb is not None:
            transforms.append(FourierEmbedding(**config.Network.fourier_emb))

        config.Network.input_transform = nn.Sequential(*transforms)

        net_model = MlpNet(**config.Network)
        self.T = config.physics.T
        self.X = config.physics.X
        self.Y = config.physics.Y
        self.nu = config.physics.nu

        self.use_causal = config.Training.Weighting.use_causal
        if self.use_causal:
            self.causal_tol = config.Training.Weighting.causal_tol
            self.causal_num = config.Training.Weighting.causal_num
            self.causal_mat = bkd.triu(bkd.ones((self.causal_num, self.causal_num)), 1).T

        super(NavierStokes2DSolver, self).__init__(config, models=net_model)


    def losses(self, batch):

        res_batch = batch["res"]
        ics_batch = batch["ics"]

        # residual loss
        res_input = res_batch['input'].to(self.config.Device)
        res_input.requires_grad_(True)
        res_pred = self.forward(inn_var=res_input)
        res_pred = self.residual(inn_var=res_input, out_var=res_pred)

        if self.use_causal:
            res_pred, _ = \
                casual_loss_weights(time_vector=res_input[..., -1],
                                    residual=res_pred,
                                    causal_num=self.causal_num,
                                    causal_mat=self.causal_mat,
                                    causal_tol=self.causal_tol)

        res_loss_m = self.loss_funcs(res_pred[..., (0,)], 0)
        res_loss_c = self.loss_funcs(res_pred[..., (1,)], 0)

        # inflow loss
        ics_input, ics_true = (ics_batch['input'].to(self.config.Device),
                               ics_batch['target'].to(self.config.Device))
        ics_input.requires_grad_(True)
        ics_pred = self.forward(inn_var=ics_input)
        ics_pred = self.output_transform(inn_var=ics_input, out_var=ics_pred)
        ics_loss_u = self.loss_funcs(ics_pred[..., (0,)], ics_true[..., (0,)])
        ics_loss_v = self.loss_funcs(ics_pred[..., (1,)], ics_true[..., (1,)])
        ics_loss_w = self.loss_funcs(ics_pred[..., (-1,)], ics_true[..., (-1,)])

        self.loss_dict.update({
            "u_ic": ics_loss_u,
            "v_ic": ics_loss_v,
            "w_ic": ics_loss_w,
            "r_m": res_loss_m,
            "r_c": res_loss_c,
        })

        return self.loss_dict

    def infer(self, batch):

        valid_batch = batch["all"]
        # residual loss
        all_input = valid_batch['input'].to(self.config.Device)
        all_input.requires_grad_(True)
        all_pred = self.forward(inn_var=all_input)
        all_pred = self.output_transform(inn_var=all_input, out_var=all_pred)
        batch["all"].update({"pred": all_pred.detach().cpu().numpy()})

        ics_batch = batch["ics"]
        # residual loss
        ics_input = ics_batch['input'].to(self.config.Device)
        ics_input.requires_grad_(True)
        ics_pred = self.forward(inn_var=ics_input)
        ics_pred = self.output_transform(inn_var=ics_input, out_var=ics_pred)
        batch["ics"].update({"pred": ics_pred.detach().cpu().numpy()})

        return batch

    def forward(self, inn_var):
        out_var = self.input_transform(inn_var)
        out_var = self.models(out_var)
        return out_var


    def residual(self, inn_var, out_var=None):

        if out_var is None:
            out_var = self.forward(inn_var)

        u = out_var[..., (0,)]
        v = out_var[..., (1,)]

        duda = gradient(u, inn_var)
        dvda = gradient(v, inn_var)

        dudx, dudy = duda[..., (0,)], duda[..., (1,)]
        dvdx, dvdy = dvda[..., (0,)], dvda[..., (1,)]

        w = dvdx - dudy
        dwda = gradient(w, inn_var)
        dwdx, dwdy, dwdt = dwda[..., (0,)], dwda[..., (1,)], dwda[..., (2,)],

        d2wdx2 = gradient(dwdx, inn_var)[..., (0,)]
        d2wdy2 = gradient(dwdy, inn_var)[..., (1,)]

        # PDE residual
        res_m = dwdt + u * dwdx + v * dwdy - (d2wdx2+d2wdy2) * self.nu
        res_c = dudx + dvdy

        return bkd.cat((res_m, res_c), dim=-1)

    def input_transform(self, inn_var):
        x = inn_var[..., (0,)]
        y = inn_var[..., (1,)]
        t = inn_var[..., (-1,)] / self.T
        return bkd.cat((x, y, t), dim=-1)

    def output_transform(self, inn_var, out_var=None):
        if out_var is None:
            out_var = self.forward(inn_var)
        u = out_var[..., (0,)]
        v = out_var[..., (1,)]
        duda = gradient(u, inn_var)
        dvda = gradient(v, inn_var)
        dudy = duda[..., (1,)]
        dvdx = dvda[..., (0,)]
        w = dvdx - dudy
        return bkd.cat((out_var, w), dim=-1)

class NavierStokes2DEvaluator(PinnEvaluator):
    def __init__(self, config, board):
        super().__init__(config, board)
        pass

    def log_plots(self, state, batch, save_fig='pred_fields'):

        real = batch['all']['target']
        pred = batch['all']['pred']
        coords = batch['all']['input']
        # draw the predicted fields
        fig, axs = plt.subplots(3, 3, num=100, figsize=(10, 8))
        self.visual.plot_fields_2D(fig, axs, real[0], pred[0], coords[0],
                                   titles=['真实', '预测', '误差'],
                                   field_names=['u', 'v', 'w'],
                                   cmaps=['jet', 'jet', 'coolwarm'])
        if isinstance(save_fig, str):
            fig.savefig(os.path.join(self.visual.save_path, save_fig + ".jpg"))
        plt.close(fig)
        self.log_dict.update({'Predicted fields at step {}'.format(coords[0, 0, 0, -1]):
                                  wandb.Image(fig2data(fig))})


        real = batch['ics']['target']
        pred = batch['ics']['pred']
        coords = batch['ics']['input']
        # draw the pde residual loss
        fig, axs = plt.subplots(3, 3, num=101, figsize=(10, 8))
        self.visual.plot_fields_2D(fig, axs, real[0], pred[0], coords[0],
                                   titles=['真实', '预测', '误差'],
                                   field_names=['u', 'v', 'w'],
                                   cmaps=['jet', 'jet', 'coolwarm'])
        if isinstance(save_fig, str):
            fig.savefig(os.path.join(self.visual.save_path, save_fig + "_ics.jpg"))
        self.log_dict.update({'Predicted ics': wandb.Image(fig2data(fig))})
        plt.close(fig)


if __name__ == "__main__":

    from config.default import get_config
    config = get_config()
    pinns = NavierStokes2DSolver(config)
    x = bkd.ones([100, 50, 2]).to(config.network.device)
    x.requires_grad_(True)

    y = pinns.forward(x)
    y = pinns.output_transform(x, y)
    res = pinns.residual(x, y)
