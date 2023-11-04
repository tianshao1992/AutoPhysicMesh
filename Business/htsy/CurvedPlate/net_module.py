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

import numpy as np
import wandb
import matplotlib.pyplot as plt

from Module import bkd, nn
from ModuleZoo.NNs.fno.FNOs import FNO
from Module.DataModule import NetFitter, NetFitterEvaluator
from Utilizes.commons import fig2data

class Heat3DPredictor(NetFitter):
    def __init__(self, config):
        # super(NavierStokes2D, self).__init__()
        models = nn.ModuleDict()
        models['reconstruct'] = FNO(**config.Network.reconstruct)
        super(Heat3DPredictor, self).__init__(config, models)
        self.config_setup(config)

    def losses(self, batch):

        loss_dict = {}
        reconstruct_batch = batch["reconstruct"]
        coords_batch = batch['coords']

        # reconstruct loss
        reconstruct_input = reconstruct_batch['input_norm'].to(self.config.Device)
        reconstruct_true = reconstruct_batch['target_norm'].to(self.config.Device)
        coords_input = coords_batch['input_norm'].to(self.config.Device)

        batch_size = reconstruct_input.shape[0]
        reconstruct_input = reconstruct_input.permute((0, 2, 3, 4, 1, 5)).squeeze(-1)
        reconstruct_true = reconstruct_true.permute((0, 2, 3, 4, 1, 5)).squeeze(-1)
        coords_input = bkd.tile(coords_input, (batch_size, 1, 1, 1, 1))

        reconstruct_pred = self.models['reconstruct'](x=reconstruct_input, grid=coords_input)
        reconstruct_loss = self.loss_funcs['reconstruct'](reconstruct_pred, reconstruct_true)

        loss_dict.update({
            "reconstruct_loss": reconstruct_loss,
        })
        return loss_dict

    def metrics(self, batch, *args, **kwargs):

        metric_dict = {}
        reconstruct_batch = batch["reconstruct"]
        # reconstruct_input = reconstruct_batch['input']
        reconstruct_true = reconstruct_batch['target'].permute((0, 2, 3, 4, 1, 5)).squeeze(-1)
        reconstruct_pred = reconstruct_batch['pred'].permute((0, 2, 3, 4, 1, 5)).squeeze(-1)

        # reconstruct metric
        reconstruct_metric = self.metric_evals.metric_funcs['reconstruct'](reconstruct_pred, reconstruct_true)
        metric_dict.update({
            "reconstruct": reconstruct_metric,
        })

        return metric_dict

    def infer(self, batch):
        with bkd.no_grad():
            reconstruct_batch = batch["reconstruct"]
            coords_batch = batch['coords']
            reconstruct_input = reconstruct_batch['input_norm'].to(self.config.Device)
            # reconstruct_true = reconstruct_batch['target_norm'].to(self.config.Device)
            coords_input = coords_batch['input_norm'].to(self.config.Device)

            batch_size = reconstruct_input.shape[0]
            reconstruct_input = reconstruct_input.permute((0, 2, 3, 4, 1, 5)).squeeze(-1)
            # reconstruct_true = reconstruct_true.permute((0, 2, 3, 4, 1, 5)).squeeze(-1)
            coords_input = bkd.tile(coords_input, (batch_size, 1, 1, 1, 1))
            reconstruct_pred = self.models['reconstruct'](x=reconstruct_input, grid=coords_input)
            reconstruct_pred = reconstruct_pred.permute((0, 4, 1, 2, 3))[..., None]
            batch["reconstruct"].update({"pred": reconstruct_pred.cpu()})

        return batch


    def pred_run(self, test_loader, evaluator):

        self.models.eval()
        for case_index, batch in enumerate(test_loader):
            batch = test_loader.batch_preprocess(batch)
            batch = self.infer_step(batch)
            batch = test_loader.batch_postprocess(batch)
            coords = batch['coords']['input'][0]

            # tecplot 输出所有时刻结果
            visual_len = batch['reconstruct']['input'].shape[0]
            reconstruct_true = batch['reconstruct']['target']
            reconstruct_pred = batch['reconstruct']['pred']

            output_len = self.config.physics.output_len
            case_path = os.path.join(evaluator.visual.save_path, "tecplot_{}".format(str(case_index)))
            if not os.path.exists(case_path):
                os.makedirs(case_path)

            # for pred_step in range(output_len):
            #     print('pred_step: {}'.format(pred_step))
            #     for visual_step in range(visual_len):
            #         evaluator.visual.output_tecplot(
            #             file=os.path.join(case_path,
            #                               "pred_step_{}_visual_step_{}.plt".format(pred_step, visual_step)),
            #             real=reconstruct_true[visual_step, pred_step] + 273.15,
            #             pred=reconstruct_pred[visual_step, pred_step] + 273.15,
            #             coord=coords,
            #             field_names=[f'T/K']) # ℃ is not so the temperature unit is transformed to K

            reconstruct_pred = reconstruct_pred.permute((0, 2, 3, 4, 1, 5)).squeeze(-1)
            reconstruct_true = reconstruct_true.permute((0, 2, 3, 4, 1, 5)).squeeze(-1)


            probes = [(4, 10, 4), (4, 6, 4), (4, 4, 4)]
            for probe in probes:
                fig, axs = plt.subplots(output_len, 2, num=100, figsize=(16, 8))
                evaluator.visual.plot_fields_1D(fig, axs,
                                                reconstruct_pred[::5, probe[0], probe[1], probe[2], :] + 273.15,
                                                reconstruct_true[::5, probe[0], probe[1], probe[2], :] + 273.15,
                                                np.arange(visual_len)[::5] * 10,
                                                titles=['真实-预测', '误差'],
                                                xylabels=('t/s', 'T/K'),
                                                field_names=['T/K'])
                fig.savefig(os.path.join(evaluator.visual.save_path,
                                         "case_{}_probe_{}".format(str(case_index), str(probe)) + ".jpg"))
                plt.close(fig)

            # evaluator.log_plots(state="test", batch=batch)



class Heat3DEvaluator(NetFitterEvaluator):
    def __init__(self, config, board):
        super(Heat3DEvaluator, self).__init__(config, board)

    def log_plots(self, state, batch):

        prefix = state.prefix_name
        output_len = self.config.physics.output_len
        if prefix == "train":
            return None

        reconstruct_batch = batch["reconstruct"]
        # reconstruct_input = reconstruct_batch['input']
        reconstruct_true = reconstruct_batch['target'].permute((0, 2, 3, 4, 1, 5)).squeeze(-1)
        reconstruct_pred = reconstruct_batch['pred'].permute((0, 2, 3, 4, 1, 5)).squeeze(-1)

        coords_batch = batch['coords']
        coords = coords_batch['input'][0]

        for time_step in range(0, reconstruct_true.shape[0], 20):

            # x-y plane
            fig, axs = plt.subplots(3, output_len, num=100+time_step, figsize=(20, 8))
            field_names = ['T/℃-' + '({})'.format(i) for i in range(output_len)]
            self.visual.plot_fields_2D(fig, axs,
                                       reconstruct_true[time_step, :, :, 4],
                                       reconstruct_pred[time_step, :, :, 4],
                                       coords[:, :, 4, :2],
                                       titles=['真实', '预测', '误差'], field_names=field_names,
                                       cmaps=['jet', 'jet', 'coolwarm'])

            save_name = prefix + "_input_step_{}_x-y".format(time_step)
            fig.savefig(os.path.join(self.visual.save_path, save_name + ".jpg"))
            self.log_dict.update({save_name: wandb.Image(fig2data(fig))})
            plt.close(fig)

            # x-z plane
            fig, axs = plt.subplots(3, output_len, num=100+time_step, figsize=(8, 8))
            self.visual.plot_fields_2D(fig, axs,
                                       reconstruct_true[time_step, :, 5, :],
                                       reconstruct_pred[time_step, :, 5, :],
                                       coords[:, 10, :, 0::2],
                                       titles=['真实', '预测', '误差'], field_names=field_names,
                                       cmaps=['jet', 'jet', 'coolwarm'])
            save_name = prefix + "_input_step_{}_x-z".format(time_step)
            fig.savefig(os.path.join(self.visual.save_path, save_name + ".jpg"))
            self.log_dict.update({save_name: wandb.Image(fig2data(fig))})
            plt.close(fig)

            # y-z plane
            fig, axs = plt.subplots(3, output_len, num=100+time_step, figsize=(10, 15))
            self.visual.plot_fields_2D(fig, axs,
                                       reconstruct_true[time_step, 4, :, :],
                                       reconstruct_pred[time_step, 4, :, :],
                                       coords[4, :, :, 1:],
                                       titles=['真实', '预测', '误差'], field_names=field_names,
                                       cmaps=['jet', 'jet', 'coolwarm'])
            save_name = prefix + "_input_step_{}_y-z".format(time_step)
            fig.savefig(os.path.join(self.visual.save_path, save_name + ".jpg"))
            self.log_dict.update({save_name: wandb.Image(fig2data(fig))})
            plt.close(fig)

