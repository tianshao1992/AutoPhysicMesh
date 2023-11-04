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
import matplotlib.pyplot as plt

from Module import bkd, nn
from ModuleZoo.NNs.mlp.MLPs import FourierEmbedding, PeriodsEmbedding, MlpNet
from Module.FusionModule import PinnSolver, PinnEvaluator
from Module.NNs.autograd import gradient
from Module.NNs.lossfuncs import get as get_loss
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
        self.Re = config.physics.Re

        super(NavierStokes2DSolver, self).__init__(config, models=net_model)



    def losses(self, batch):

        loss_dict = {}
        res_batch = batch["res"]
        bcs_batch = batch["bcs"]

        # residual loss
        res_input = res_batch['input'].to(self.config.Device)
        res_input.requires_grad_(True)
        res_pred = self.models(res_input)
        res_pred = self.residual(inn_var=res_input, out_var=res_pred)
        res_loss_x = self.loss_funcs(res_pred[..., (0,)], 0)
        res_loss_y = self.loss_funcs(res_pred[..., (1,)], 0)
        res_loss_c = self.loss_funcs(res_pred[..., (2,)], 0)

        # inflow loss
        bcs_input, bcs_true = (bcs_batch['input'].to(self.config.Device),
                               bcs_batch['target'].to(self.config.Device))
        # bcs_input.requires_grad_(True)
        bcs_pred = self.models(bcs_input)
        bcs_loss_u = self.loss_funcs(bcs_pred[..., (1,)], bcs_true[..., (1,)])
        bcs_loss_v = self.loss_funcs(bcs_pred[..., (2,)], bcs_true[..., (2,)])


        loss_dict.update({
            "u_bc": bcs_loss_u,
            "v_bc": bcs_loss_v,
            "r_x": res_loss_x,
            "r_y": res_loss_y,
            "r_c": res_loss_c,
        })

        return loss_dict


    def infer(self, batch):

        valid_batch = batch["all"]
        # residual loss
        all_input = valid_batch['input'].to(self.config.Device)
        all_input.requires_grad_(True)
        all_pred = self.models(all_input)
        all_res = self.residual(inn_var=all_input, out_var=all_pred)
        batch["all"].update({"pred": all_pred.detach().cpu()})
        batch["all"].update({"residual": all_res.detach().cpu()})
        return batch

    def metrics(self, batch, *args, **kwargs):

        metric_dict = {}
        valid_batch = batch["all"]
        residual = valid_batch['residual']
        pred = valid_batch['pred']
        true = valid_batch['target']

        # residual metric
        residual_metric = self.metric_evals.metric_funcs['residual'](residual, 0)

        # fields metric
        # the first channel is pressure, the rest are velocity,
        # so we need to remove the first channel, due to the pressure is all zero in the true.
        fields_metric = self.metric_evals.metric_funcs['fields'](pred[..., 1:], true[..., 1:])

        metric_dict.update({
            "residual": residual_metric,
            "fields": fields_metric,
        })

        return metric_dict


    def residual(self, inn_var, out_var=None):

        if out_var is None:
            out_var = self.forward(inn_var)

        p = out_var[..., (0,)]
        u = out_var[..., (1,)]
        v = out_var[..., (2,)]

        dpda = gradient(p, inn_var)
        duda = gradient(u, inn_var)
        dvda = gradient(v, inn_var)

        dpdx, dpdy = dpda[..., (0,)], dpda[..., (1,)]
        dudx, dudy = duda[..., (0,)], duda[..., (1,)]
        dvdx, dvdy = dvda[..., (0,)], dvda[..., (1,)]

        d2udx2 = gradient(dudx, inn_var)[..., (0,)]
        d2udy2 = gradient(dudy, inn_var)[..., (1,)]
        d2vdx2 = gradient(dvdx, inn_var)[..., (0,)]
        d2vdy2 = gradient(dvdy, inn_var)[..., (1,)]

        # PDE residual
        res_x = u * dudx + v * dudy + dpdx - (d2udx2+d2udy2) / self.Re
        res_y = u * dvdx + v * dvdy + dpdy - (d2vdx2+d2vdy2) / self.Re
        res_c = dudx + dvdy

        return bkd.cat((res_x, res_y, res_c), dim=-1)


class NavierStokes2DEvaluator(PinnEvaluator):
    def __init__(self, config, board):
        super().__init__(config, board)
        pass

    def log_plots(self, state, batch, save_fig='pred_fields'):

        real = batch['all']['target'][0]
        pred = batch['all']['pred'][0]
        coords = batch['all']['input'][0]
        residual = batch['all']['residual'][0]
        # draw the predicted fields
        fig, axs = plt.subplots(3, 3, num=100, figsize=(10, 8))
        self.visual.plot_fields_2D(fig, axs, real, pred, coords,
                                   titles=['真实', '预测', '误差'],
                                   field_names=['p', 'u', 'v'],
                                   cmaps=['jet', 'jet', 'coolwarm'])

        fig.savefig(os.path.join(self.visual.save_path, "pred_fields.jpg"))
        plt.close(fig)
        self.log_dict.update({'Predicted fields': self.board.Image(fig2data(fig))})


        fig, axs = plt.subplots(3, 1, num=100, figsize=(3, 8))
        self.visual.plot_fields_2D(fig, axs, residual, None, coords,
                                   titles=['真实', ],
                                   field_names=['动量x', '动量y', '连续性'],
                                   cmaps=['jet'])

        fig.savefig(os.path.join(self.visual.save_path, "pred_residual.jpg"))
        plt.close(fig)
        self.log_dict.update({'Predicted residual': self.board.Image(fig2data(fig))})


        fig, axs = plt.subplots(3, 2, num=100, figsize=(8, 8))
        self.visual.plot_fields_1D(fig, axs, real[220, :], pred[220, :], coords[128, :, 0],
                                   titles=['真实', '误差'], xylabels=('y', 'fields'),
                                   field_names=['压力', '速度x', '速度y'])

        fig.savefig(os.path.join(self.visual.save_path, "pred_middle.jpg"))
        plt.close(fig)
        self.log_dict.update({'Predicted middle line': self.board.Image(fig2data(fig))})

