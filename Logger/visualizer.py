#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
# @Copyright (c) 2023 Baidu.com, Inc. All Rights Reserved
# @Time    : 2023/10/18 14:30
# @Author  : Liu Tianyuan (liutianyuan02@baidu.com)
# @File    : visualizer.py
# @Description    : ******
"""
import os
import logging
import sys
import pandas as pd
import numpy as np
import seaborn as sbn
from scipy import stats
import matplotlib as mpl
import matplotlib.tri as tri
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib import ticker, rcParams, font_manager
from matplotlib.ticker import MultipleLocator
from matplotlib.animation import FuncAnimation
from mpl_toolkits.axes_grid1 import make_axes_locatable
# mpl.use('Agg')


class Visual(object):
    """
    log文件记录所有打印结果
    """

    def __init__(self,
                 save_path='./work/visuals',
                 use_tex='ch-en',
                 font_size=12,
                 font_file='TNW_SIMSUN.TTF'):
        self.use_tex = use_tex
        self.font_size = font_size
        self.font_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), font_file)
        self._setup()
        self.save_path = save_path
        if not os.path.exists(save_path):
            os.makedirs(save_path)


    def _setup(self):
        config = {
            "font.family": 'serif',
            "font.size": self.font_size,
            "mathtext.fontset": 'stix',
            "font.serif": ['Times New Roman'],
            "figure.constrained_layout.use": True,  # solve the problem that text is covered by the other text
            "figure.dpi": 300,
            # "figure.figsize": 300,
            "axes.titlesize": 12,
            "axes.labelsize": 12,
            # "axes.linewidth ": 1.0,
            "axes.unicode_minus": False,
            "xtick.labelsize": 12,
            "ytick.labelsize": 12,
            "lines.linewidth": 3.0,
        }

        if 'ch' in self.use_tex.lower():
            ########################中英文混合方案一：采用中英文混合字体################################
            #  https://zhuanlan.zhihu.com/p/501395717
            # 使用简单，但是无法显示公式中的中文字符
            # 字体加载

            font_manager.fontManager.addfont(self.font_path)
            prop = font_manager.FontProperties(fname=self.font_path)
            # 字体设置
            config['font.family'] = 'sans-serif'  # 使用字体中的无衬线体
            config['font.sans-serif'] = prop.get_name()  # 根据名称设置字体

            ########################中英文混合方案二：主要为中文，而英文放入latex公式################################
            # #  https://zhuanlan.zhihu.com/p/118601703
            # # 使用内置tex,在使用英文时需要设置在公式中，且由于公式默认斜体，则正体需要使用\mathrm{text}

            #
            # ########################中英文混合方案三：调用xelatex################################
            # #  https://zhuanlan.zhihu.com/p/118601703
            # # 外部的tex渲染
            # matplotlib.use("pgf")
            # pgf_config = {
            #     "font.family": 'serif',
            #     "font.size": 20,
            #     "pgf.rcfonts": False,
            #     "text.usetex": True,
            #     "pgf.preamble": [
            #         r"\usepackage{unicode-math}",
            #         # r"\setmathfont{XITS Math}",
            #         # 这里注释掉了公式的XITS字体，可以自行修改
            #         r"\setmainfont{Times New Roman}",
            #         r"\usepackage{xeCJK}",
            #         r"\xeCJKsetup{CJKmath=true}",
            #         r"\setCJKmainfont{SimSun}",
            #     ],
            # }
            # rcParams.update(pgf_config)


            rcParams.update(config)

    def plot_fields_2D(self, fig, axs, real, pred=None, coord=None,
                       edges=None, mask=None,
                       cmin_max=None, fmin_max=None,
                       show_channel=None, field_names=None, cmaps=None, titles=None):
        '''

        :param fig: the fig of matplotlib
        :param axs: the axs of fig: ndarray
        :param real: the real fields: ndarray [batch, num_of_fields]
        :param pred: the predicted fields: ndarray [batch, num_of_fields]
        :param coord: the predicted fields: ndarray [batch, num_of_coords]
        :param edges: the link of edges,
        :param mask: the hole profile in spatial
        :param cmin_max:
        :param fmin_max:
        :param show_channel: the index of channel in origin data
        :param field_names: the name of channel: a list of string, for example, ['p', 't', 'u', 'v']
        :param cmaps:
        :param titles: the name of width: a list of string, ['truth', 'pred', 'error'] or ['truth', ]
        :return:
        '''

        if len(real.shape) == 3:
            graph_flag = False
        else:
            graph_flag = True

        if len(axs.shape) == 1:
            if pred is None:
                axs = axs[:, None]
            else:
                axs = axs[None, :]

        if show_channel is None:
            show_channel = np.arange(real.shape[-1])

        if field_names is None:
            field_names= []
            for i in show_channel:
                field_names.append('field ' + str(i+1))

        if fmin_max is None:
            faxis = tuple(range(len(real.shape) - 1))
            fmin, fmax = real.min(axis=faxis), real.max(axis=faxis)
        else:
            fmin, fmax = fmin_max[0], fmin_max[1]

        if cmin_max is None:
            caxis = tuple(range(len(coord.shape)-1))
            cmin, cmax = coord.min(axis=caxis), coord.max(axis=caxis)
        else:
            cmin, cmax = cmin_max[0], cmin_max[1]

        if titles is None:
            titles = ['truth', 'predicted', 'error']

        if cmaps is None:
            cmaps = ['RdYlBu_r', 'RdYlBu_r', 'coolwarm']

        if graph_flag and edges is None:
            triang = tri.Triangulation(coord[:, 0], coord[:, 1])
            edges = triang.edges


        x_pos = coord[..., 0]
        y_pos = coord[..., 1]
        size_channel = len(show_channel)

        if pred is None:
            show_width = 1
        else:
            show_width = 3
        for i in range(size_channel):
            fi = show_channel[i]
            if pred is None:
                ff = [real[..., fi], ]
            else:
                ff = [real[..., fi], pred[..., fi], real[..., fi] - pred[..., fi]]

            limit = max(abs(ff[-1].min()), abs(ff[-1].max()))
            for j in range(show_width):

                if graph_flag:
                    f_true = axs[i][j].tripcolor(x_pos, y_pos, ff[j],
                                                 triangles=edges, cmap=cmaps[j], shading='gouraud',
                                                 antialiased=True, snap=True)
                else:
                    f_true = axs[i][j].pcolormesh(x_pos, y_pos, ff[j], cmap=cmaps[j], shading='gouraud',
                                                  antialiased=True, snap=True)

                # f_true = axs[i][j].tricontourf(triObj, ff[j], 20, cmap=cmaps[j])
                if mask is not None:
                    axs[i][j].fill(mask[:, 0], mask[:, 1], facecolor='white')
                # f_true.set_zorder(10)

                # axs[i][j].grid(zorder=0, which='both', color='grey', linewidth=1)
                axs[i][j].set_title(titles[j])
                axs[i][j].axis([cmin[0], cmax[0], cmin[1], cmax[1]])
                cb = fig.colorbar(f_true, ax=axs[i][j])
                # cb.ax.tick_params(labelsize=15)
                # tick_locator = ticker.MaxNLocator(nbins=6)  # colorbar上的刻度值个数
                # cb.locator = tick_locator
                cb.update_ticks()
                if j < 2:
                    f_true.set_clim(fmin[i], fmax[i])
                    cb.ax.set_title(field_names[i], loc='center')
                else:
                    f_true.set_clim(-limit, limit)
                    cb.ax.set_title('$\mathrm{\Delta}$' + field_names[i], loc='center')
                # 设置刻度间隔
                axs[i][j].set_aspect(1)
                axs[i][j].set_xlabel('x')
                axs[i][j].set_ylabel('y')



if __name__=='__main__':
    visual_model = Visual(use_tex='ch-en')

    # random data
    x = np.random.rand(10, 1)
    y = np.random.rand(10, 1)
    coord = np.concatenate([x, y], axis=-1)
    real = np.random.rand(10, 1)
    pred = np.random.rand(10, 1)
    mask = np.array([[0, 0], [0, 0.5], [0.5, 0.5], [0.5, 0]])
    fig, axs = plt.subplots(1, 3)
    visual_model.plot_fields_tr(fig, axs, real, pred, coord, mask=mask, titles=['真实field', '预测field', '误差field'])
    a = 1
    plt.show()  # fig.show();input();

    # airfoil reading from msh & random fields
    # import meshio
    data_file = '../demo/cylinder2d+t/data/ns_unsteady.npy'
    data = np.load(data_file, allow_pickle=True).item()
    u_ref = np.array(data["u"], dtype=np.float32)
    v_ref = np.array(data["v"], dtype=np.float32)
    p_ref = np.array(data["p"], dtype=np.float32)
    coords = np.array(data["coords"], dtype=np.float32)
    cylinder_coords = np.array(data["cylinder_coords"], dtype=np.float32)

    real = np.stack((p_ref[0], u_ref[0], v_ref[0]), axis=1)
    fig, axs = plt.subplots(3, 1)
    visual_model.plot_fields_tr(fig, axs, real, None, coords, mask=cylinder_coords,
                                titles=['真实field', '预测field', '误差field'], field_names=['p', 'u', 'v'])
    a = 1
    plt.show()  # fig.show();input();





