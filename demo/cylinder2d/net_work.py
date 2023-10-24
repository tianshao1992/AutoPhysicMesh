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
from NetZoo.mlp.MLPs import FourierEmbedding, MlpNet
from Module.pinn import BasicSolver, BaseEvaluator
from Module.autograd import jacobian, gradient
from Module.lossfuncs import get as get_loss
from Utilizes.commons import fig2data
loss_func = get_loss('mse')

class NavierStokes2DSolver(BasicSolver):
    def __init__(self, config):
        # super(NavierStokes2D, self).__init__()

        self.config_setup(config)
        self.loss_dict = {}
        self.ntk_dict = {}
        self.total_loss = 1e20

    def forward(self, inn_var):
        out_var = self.input_transform(inn_var)
        out_var = self.net_model(out_var)
        out_var = self.output_transform(inn_var, out_var)
        return out_var

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

        # outflow boundary residual
        u_out = dudx / self.Re - p
        v_out = dvdx

        return bkd.cat((res_x, res_y, res_c), dim=-1), bkd.cat((u_out, v_out), dim=-1)

    def res_and_w(self, inn_var):
        # Sort temporal coordinates
        index_sorted = inn_var[..., -1].argsort()
        pred_pde, _ = self.residual(inn_var[index_sorted])

        ru_pred = pred_pde[..., 0].reshape(self.num_chunks, -1)
        rv_pred = pred_pde[..., 1].reshape(self.num_chunks, -1)
        rc_pred = pred_pde[..., 2].reshape(self.num_chunks, -1)

        ru_l = bkd.mean(bkd.square(ru_pred), dim=1)
        rv_l = bkd.mean(bkd.square(rv_pred), dim=1)
        rc_l = bkd.mean(bkd.square(rc_pred), dim=1)

        ru_gamma = bkd.exp(-self.tol * (self.M.to(inn_var) @ ru_l)).detach()
        rv_gamma = bkd.exp(-self.tol * (self.M.to(inn_var) @ rv_l)).detach()
        rc_gamma = bkd.exp(-self.tol * (self.M.to(inn_var) @ rc_l)).detach()

        # Take minimum of the causal weights
        gamma = bkd.vstack([ru_gamma, rv_gamma, rc_gamma])
        gamma = bkd.min(gamma, dim=0)[0]
        ru_loss = bkd.mean(gamma * ru_l)
        rv_loss = bkd.mean(gamma * rv_l)
        rc_loss = bkd.mean(gamma * rc_l)
        return [ru_loss, rv_loss, rc_loss], gamma

    def losses(self, batch):

        ics_batch = batch["ics"]
        inflow_batch = batch["inflow"]
        outflow_batch = batch["outflow"]
        wall_batch = batch["wall"]
        cylinder_batch = batch["cylinder"]
        res_batch = batch["res"]

        # residual loss
        res_input = res_batch['input']
        res_input.requires_grad_(True)
        if self.use_causal:
            [res_loss_x, res_loss_y, res_loss_c], _ = self.res_and_w(inn_var=res_input)
        else:
            res_pred = self.forward(inn_var=res_input)
            res_pred, _ = self.residual(inn_var=res_input, out_var=res_pred)
            res_loss_x = loss_func(res_pred[..., (0,)], 0)
            res_loss_y = loss_func(res_pred[..., (1,)], 0)
            res_loss_c = loss_func(res_pred[..., (2,)], 0)

        # initial condition loss
        ics_input, ics_true = ics_batch['input'], ics_batch['target']
        ics_pred = self.forward(inn_var=ics_input)
        ics_loss_p = loss_func(ics_pred[..., (0,)], ics_true[..., (0,)])
        ics_loss_u = loss_func(ics_pred[..., (1,)], ics_true[..., (1,)])
        ics_loss_v = loss_func(ics_pred[..., (2,)], ics_true[..., (2,)])

        # inflow loss
        inflow_input, inflow_true = inflow_batch['input'], inflow_batch['target']
        inflow_pred = self.forward(inn_var=inflow_input)
        inflow_loss_u = loss_func(inflow_pred[..., (1,)], inflow_true[..., (1,)])
        inflow_loss_v = loss_func(inflow_pred[..., (2,)], 0)

        # outflow loss
        outflow_input = outflow_batch['input']
        outflow_input.requires_grad_(True)
        outflow_pred = self.forward(inn_var=outflow_input)
        _, outflow_res = self.residual(outflow_input, outflow_pred)
        outflow_loss_u = loss_func(outflow_res[..., (0,)], 0)
        outflow_loss_v = loss_func(outflow_res[..., (1,)], 0)

        # wall loss
        wall_input = wall_batch['input']
        wall_pred = self.forward(inn_var=wall_input)
        wall_loss_u = loss_func(wall_pred[..., (1,)], 0)
        wall_loss_v = loss_func(wall_pred[..., (2,)], 0)

        # wall loss
        cylinder_input = cylinder_batch['input']
        cylinder_pred = self.forward(inn_var=cylinder_input)
        cylinder_loss_u = loss_func(cylinder_pred[..., (1,)], 0)
        cylinder_loss_v = loss_func(cylinder_pred[..., (2,)], 0)


        self.loss_dict.update({
            "p_ic": ics_loss_p,
            "u_ic": ics_loss_u,
            "v_ic": ics_loss_v,
            "u_in": inflow_loss_u,
            "v_in": inflow_loss_v,
            "u_out": outflow_loss_u,
            "v_out": outflow_loss_v,
            "u_wall": wall_loss_u,
            "v_wall": wall_loss_v,
            "u_cylinder": cylinder_loss_u,
            "v_cylinder": cylinder_loss_v,
            "r_x": res_loss_x,
            "r_y": res_loss_y,
            "r_c": res_loss_c,
        })

        return self.loss_dict

    def config_setup(self, config):

        if config.network.fourier_emb is not None:
            config.network.input_transform = FourierEmbedding(**config.network.fourier_emb)

        self.net_model = MlpNet(**config.network)

        super().config_setup(config)

        # Non-dimensionalized domain length and width
        self.L, self.W, self.T = config.physics.L, config.physics.W, config.physics.T
        self.Re = config.physics.Re  # Reynolds number

        self.U_star = config.physics.U_star
        self.L_star = config.physics.L_star
        self.use_causal = config.weighting.use_causal

        if config.weighting.use_causal:
            self.tol = config.weighting.causal_tol
            self.num_chunks = config.weighting.num_chunks
            self.M = bkd.triu(bkd.ones((self.num_chunks, self.num_chunks)), 1).T


    def input_transform(self, inn_var):
        x = inn_var[..., (0,)] / self.L
        y = inn_var[..., (1,)] / self.W
        t = inn_var[..., (-1,)] / self.T
        return bkd.cat((x, y, t), dim=-1)

    def output_transform(self, inn_var, out_var):
        y_hat = inn_var[..., (1,)] * self.L_star * self.W
        out_var[..., (1,)] = out_var[..., (1,)] + 4 * 1.5 * y_hat * (0.41 - y_hat) / (0.41**2)
        return out_var

class NavierStokes2DEvaluator(BaseEvaluator):
    def __init__(self, config, model):
        super().__init__(config, model)
        pass


    def eval_res(self, batch):
        self.module.net_model.eval()
        ics_batch = batch["ics"]
        inflow_batch = batch["inflow"]
        outflow_batch = batch["outflow"]
        wall_batch = batch["wall"]
        cylinder_batch = batch["cylinder"]
        res_batch = batch["res"]

        # residual loss
        res_input = res_batch['input']
        res_input.requires_grad_(True)
        res_pred = self.module.forward(inn_var=res_input)
        res_res, _ = self.module.residual(inn_var=res_input, out_var=res_pred)

        # initial condition loss
        ics_input, ics_true = ics_batch['input'], ics_batch['target']
        ics_pred = self.module.forward(inn_var=ics_input)
        # inflow loss
        inflow_input, inflow_true = inflow_batch['input'], inflow_batch['target']
        inflow_pred = self.module.forward(inn_var=inflow_input)
        # outflow loss
        outflow_input = outflow_batch['input']
        outflow_input.requires_grad_(True)
        outflow_pred = self.module.forward(inn_var=outflow_input)
        _, outflow_res = self.module.residual(outflow_input, outflow_pred)
        # wall loss
        wall_input = wall_batch['input']
        wall_pred = self.module.forward(inn_var=wall_input)
        # wall loss
        cylinder_input = cylinder_batch['input']
        cylinder_pred = self.module.forward(inn_var=cylinder_input)

        pred_res ={
            "res": {'input': res_input.detach().cpu().numpy(), 'pred': res_pred.detach().cpu().numpy(),
                    'target': res_batch['output'].detach().cpu().numpy(), 'residual': res_res.detach().cpu().numpy()},
            "ics": {'input': ics_input.detach().cpu().numpy(), 'pred': ics_pred.detach().cpu().numpy(),
                    'target': ics_true.detach().cpu().numpy(), },
            "inflow": {'input': inflow_input.detach().cpu().numpy(), 'pred': inflow_pred.detach().cpu().numpy(),
                    'target': inflow_true.detach().cpu().numpy(),},
            "outflow": {'input': outflow_input.detach().cpu().numpy(), 'pred': outflow_pred.detach().cpu().numpy(),
                    'target': outflow_batch['output'].detach().cpu().numpy(), 'residual': outflow_res.detach().cpu().numpy()},
            "wall": {'input': wall_input.detach().cpu().numpy(), 'pred': wall_pred.detach().cpu().numpy(),
                    'target': wall_batch['output'].detach().cpu().numpy(),},
            "cylinder": {'input': cylinder_input.detach().cpu().numpy(), 'pred': cylinder_pred.detach().cpu().numpy(),
                    'target': cylinder_batch['output'].detach().cpu().numpy(),},
        }
        return pred_res

    def log_plot(self, batch, save_fig='pred_fields'):
        pred_res = self.eval_res(batch)
        real = pred_res['res']['pred']
        coords = pred_res['res']['input']
        fig, axs = plt.subplots(3, 1, num=100, figsize=(10, 8))
        self.visual.plot_fields_tr(fig, axs, real, None, coords,
                                   titles=['预测field',], field_names=['p', 'u', 'v'],
                                   cmaps=['jet', 'jet', 'coolwarm'])
        if isinstance(save_fig, str):
            fig.savefig(os.path.join(self.visual.save_path, save_fig + ".jpg"))
        self.log_dict.update({'Predicted fields':
                                  wandb.Image(fig2data(fig))})
        plt.close(fig)



if __name__ == "__main__":

    from all_config import get_config
    config = get_config()
    pinn_cyl = NavierStokes2DSolver(config)
    x = bkd.ones([100, 50, 3])
    x.requires_grad_(True)
    y = pinn_cyl(x)
    print(y.shape)
    xde, _ = pinn_cyl.residual(x, y)

    print(xde.shape)
