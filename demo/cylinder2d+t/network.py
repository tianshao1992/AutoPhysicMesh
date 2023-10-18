#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
# @Copyright (c) 2023 Baidu.com, Inc. All Rights Reserved
# @Time    : 2023/10/17 0:15
# @Author  : Liu Tianyuan (liutianyuan02@baidu.com)
# @File    : network.py
# @Description    : ******
"""
from model import bkd, nn
from model.networks import FourierEmbedding, MlpNet
from model.pinn import BasicModule
from model.autograd import jacobian
from model.lossfuncs import get as get_loss

loss_func = get_loss('mse')

class NavierStokes2D(BasicModule):
    def __init__(self, config):
        # super(NavierStokes2D, self).__init__()

        self.config_setup(config)
        self.loss_dict = {}
        self.ntk_dict = {}

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

        dpda = jacobian(p, inn_var)
        duda = jacobian(u, inn_var)
        dvda = jacobian(v, inn_var)

        dpdx, dpdy = dpda[..., (0,)], dpda[..., (1,)]
        dudx, dudy = duda[..., (0,)], duda[..., (1,)]
        dvdx, dvdy = dvda[..., (0,)], dvda[..., (1,)]

        d2udx2 = jacobian(dudx, inn_var)[..., (0,)]
        d2udy2 = jacobian(dudy, inn_var)[..., (1,)]
        d2vdx2 = jacobian(dvdx, inn_var)[..., (0,)]
        d2vdy2 = jacobian(dvdy, inn_var)[..., (1,)]

        # PDE residual
        res_x = u * dudx + v * dudy + dpdx - (d2udx2+d2udy2) / self.Re
        res_y = u * dvdx + v * dvdy + dpdy - (d2vdx2+d2vdy2) / self.Re
        res_c = dudx + dvdy

        # outflow boundary residual
        u_out = dudx / self.Re - p
        v_out = dvdx

        return bkd.cat((res_x, res_y, res_c), dim=-1), bkd.cat((u_out, v_out), dim=-1)

    def losses(self, batch):

        ic_batch = batch["ic"]
        inflow_batch = batch["inflow"]
        outflow_batch = batch["outflow"]
        noslip_batch = batch["noslip"]
        res_batch = batch["res"]

        # residual loss
        res_input = res_batch['input']
        res_input.requires_grad_(True)
        res_pred = self.forward(inn_var=res_input)
        res_pred, _ = self.residual(inn_var=res_input, out_var=res_pred)
        res_loss_x = loss_func(res_pred[..., (0,)], 0)
        res_loss_y = loss_func(res_pred[..., (1,)], 0)
        res_loss_c = loss_func(res_pred[..., (2,)], 0)

        # initial condition loss
        ic_input, ic_true = ic_batch['input'], ic_batch['target']
        ic_pred = self.forward(inn_var=ic_input)
        ic_loss_p = loss_func(ic_pred[..., (0,)], ic_true[..., (0,)])
        ic_loss_u = loss_func(ic_pred[..., (1,)], ic_true[..., (1,)])
        ic_loss_v = loss_func(ic_pred[..., (2,)], ic_true[..., (2,)])

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

        # noslip loss
        noslip_input = noslip_batch['input']
        noslip_pred = self.forward(inn_var=noslip_input)
        # noslip_true = ic_batch['target']
        noslip_loss_u = loss_func(noslip_pred[..., (1,)], 0)
        noslip_loss_v = loss_func(noslip_pred[..., (2,)], 0)


        loss_dict = {
            "p_ic": ic_loss_p,
            "u_ic": ic_loss_u,
            "v_ic": ic_loss_v,
            "u_in": inflow_loss_u,
            "v_in": inflow_loss_v,
            "u_out": outflow_loss_u,
            "v_out": outflow_loss_v,
            "u_noslip": noslip_loss_u,
            "v_noslip": noslip_loss_v,
            "r_x": res_loss_x,
            "r_y": res_loss_y,
            "r_c": res_loss_c,
        }

        return loss_dict

    def config_setup(self, config):

        if config.network.fourier_emb is not None:
            config.network.input_transform = FourierEmbedding(**config.network.fourier_emb)

        self.net_model = MlpNet(**config.network)
        self.loss_weights = dict(config.weighting.init_weights)
        # Non-dimensionalized domain length and width
        self.L, self.W, self.T = config.physics.L, config.physics.W, config.physics.T
        self.Re = config.physics.Re  # Reynolds number

        if config.nondim == True:
            self.U_star = 1.0
            self.L_star = 0.1
        else:
            self.U_star = 1.0
            self.L_star = 1.0

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
        y_hat = inn_var[..., (2,)] * self.L_star * self.W
        out_var[..., (1,)] = out_var[..., (1,)] + 4 * 1.5 * y_hat * (0.41 - y_hat) / (0.41**2)
        return out_var


if __name__ == "__main__":

    from config import get_config
    config = get_config()
    pinn_cyl = NavierStokes2D(config)
    x = bkd.ones([100, 50, 3])
    x.requires_grad_(True)
    y = pinn_cyl(x)
    print(y.shape)
    xde, _ = pinn_cyl.residual(x, y)

    print(xde.shape)
