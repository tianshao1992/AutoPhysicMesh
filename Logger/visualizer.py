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

from Module.utils import tensor2numpy


def adjacent_values(vals, q1, q3):
    """
    生成四分点，plot_violin
    """
    upper_adjacent_value = q3 + (q3 - q1) * 1.5
    upper_adjacent_value = np.clip(upper_adjacent_value, q3, vals[-1])

    lower_adjacent_value = q1 - (q3 - q1) * 1.5
    lower_adjacent_value = np.clip(lower_adjacent_value, vals[0], q1)
    return lower_adjacent_value, upper_adjacent_value


def set_axis_style(ax, labels, position=None):
    """
    生成四分点，plot_violin
    """
    ax.get_xaxis().set_tick_params(direction='out')
    ax.xaxis.set_ticks_position('bottom')
    if position is None:
        ax.set_xticks(np.arange(1, len(labels) + 1))
        ax.set_xlim(0.25, len(labels) + 0.75)
    else:
        ax.set_xticks(position)
        ax.set_xlim(1.5 * position[0] - 0.5 * position[1], 1.5 * position[-1] - 0.5 * position[-2])
    ax.set_xticklabels(labels)


class Visual(object):
    """
    log文件记录所有打印结果
    """

    def __init__(self,
                 config,
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

    def plot_scatter(self, fig, axs, x, y,
                     title=None, axis_log=(False, False), xylabels=('x', 'y')):
        r"""
            plot the scatter of x and y
            Args:
                :param fig:
                :param axs:
                :param true:
                :param pred:
                :param title:
                :param axis_log:
                :param xylabels:
            :return:
                None
        """
        # sbn.set(color_codes=True)

        axs.scatter(np.arange(x.shape[0]), x, marker='*')
        axs.scatter(np.arange(y.shape[0]), y, marker='.')

        if axis_log[0]:
            axs.semilogx()
        if axis_log[1]:
            axs.semilogy()

        axs.grid(True)  # 添加网格
        axs.legend(loc="best")
        axs.set_xlabel(xylabels[0])
        axs.set_ylabel(xylabels[1])
        axs.tick_params('both')
        axs.set_title(title)

    def plot_regression(self, fig, axs, true, pred,
                        error_ratio=0.05, title=None,
                        axis_log=(False, False),
                        xylabels=('true value', 'pred value')):
        r"""
            plot the regression of true and pred
            Args:
                :param fig:
                :param axs:
                :param true:
                :param pred:
                :param error_ratio:
                :param title:
                :param axis_log:
                :param xylabels:
            :return:
                None
        """
        # sbn.set(color_codes=True)

        real = tensor2numpy(true)
        pred = tensor2numpy(pred)

        max_value = max(true)  # math.ceil(max(true)/100)*100
        min_value = min(true)  # math.floor(min(true)/100)*100
        split_value = np.linspace(min_value, max_value, 11)

        split_dict = {}
        split_label = np.zeros(len(true), np.int32)
        for i in range(len(split_value)):
            split_dict[i] = str(split_value[i])
            index = true >= split_value[i]
            split_label[index] = i + 1

        axs.scatter(true, pred, marker='.', color='firebrick', linewidth=2.0)
        axs.plot([min_value, max_value], [min_value, max_value], '-', color='steelblue', linewidth=5.0)
        # 在两个曲线之间填充颜色
        axs.fill_between([min_value, max_value], [(1 - error_ratio) * min_value, (1 - error_ratio) * max_value],
                         [((1 + error_ratio)) * min_value, ((1 + error_ratio)) * max_value],
                         alpha=0.2, color='steelblue')

        if axis_log[0]:
            axs.semilogx()
        if axis_log[1]:
            axs.semilogy()

        axs.set_xlim((min_value, max_value))
        # axs.set_ylim((min_value, max_value))
        axs.grid(True)  # 添加网格
        axs.set_xlabel(xylabels[0])
        axs.set_ylabel(xylabels[1])
        axs.tick_params('both')
        axs.set_title(title)
        axs.legend(['真实-预测', 'y=x', '±{:.2f}%'.format(error_ratio * 100)])

    def plot_error(self, fig, axs, error, error_ratio=0.05, title=None, rel_error=False,
                   xylabels=('predicted error / %', 'distribution density')):
        # sbn.set_color_codes()
        # ax.bar(np.arange(len(error)), error*100, )

        if rel_error:
            error = pd.DataFrame(error) * 100  # 转换格式
            acc = (np.abs(np.array(error)) < error_ratio * 100).sum() / error.shape[0]
        else:
            error = pd.DataFrame(error)
            acc = (np.abs(np.array(error)) < error_ratio).sum() / error.shape[0]

        # 绘制针对单变量的分布图
        sbn.distplot(error, bins=20, norm_hist=True, rug=True, fit=stats.norm, kde=False,
                     rug_kws={"color": "forestgreen"},
                     fit_kws={"color": "firebrick", "lw": 3},
                     hist_kws={"color": "steelblue"},
                     ax=axs)
        # plt.xlim([-1, 1])
        if title is None:
            if rel_error:
                title = '平均误差小于 {:.2f}% \n 占比为{:.2f}%'.format(error_ratio * 100, acc * 100)
            else:
                title = '平均误差小于 {:.2f} \n 占比为{:.2f}%'.format(error_ratio, acc * 100)

        axs.grid(True)  # 添加网格
        # axs.legend(loc="best", prop=self.font)
        axs.set_xlabel(xylabels[0])
        axs.set_ylabel(xylabels[1])
        axs.tick_params('both')
        axs.set_title(title)

    def plot_box(self, fig, ax, data,
                 title=None,
                 legends=None,
                 xylabels=None,
                 xticks=None,
                 bag_width=1.0):

        r"""
            plot the box of data
            Args:
                :param fig:
                :param ax:
                :param data:
                :param title:
                :param legends:
                :param xlabel:
                :param xticks:
                :param bag_width:
            :return:
                None
        """

        # 绘制箱形图
        ax.set_title(title)
        ax.semilogy()
        ax.grid()
        n_vin = data.shape[-1]
        colors_map = ['#E4DACE', '#E5BB4B', '#498EAF', '#631F16']
        if len(data.shape) == 2:
            positions = np.arange(n_vin) + 1
            x_pos = None
            n_bag = 1
        else:
            n_bag = data.shape[-2]
            p = (np.linspace(0, 1, n_vin + 2) - 0.5) * bag_width
            positions = np.hstack([p[1:-1] + 0.5 + i for i in range(n_bag)]) * n_vin
            x_pos = np.arange(n_bag) * n_vin + n_vin / 2
        # parts = ax.boxplot(data.reshape(data.shape[0], -1), widths=0.5 * bag_width, positions=positions, vert=True,
        #                    patch_artist=True, )
        parts = ax.boxplot(data.reshape(data.shape[0], -1).T,
                           widths=0.5 * bag_width, positions=positions,
                           vert=True, patch_artist=True, )

        for i in range(n_vin):
            for j in range(n_bag):
                parts['boxes'][i + j * n_vin].set_facecolor(colors_map[i % len(colors_map)])  # violin color
                parts['boxes'][i + j * n_vin].set_edgecolor('grey')  # violin edge
                parts['boxes'][i + j * n_vin].set_alpha(0.9)
        if legends is not None:
            ax.legend(legends)
        if xticks is None:
            xticks = np.arange(n_vin * n_bag)
        ax.set_xlabel(xylabels[0])
        ax.set_xlabel(xylabels[1])
        set_axis_style(ax, xticks, x_pos)

    def plot_violin(self, fig, ax, data, title=None, legends=None, xticks=None, xlabel=None, bag_width=1.0):

        ax.set_title(title)
        ax.semilogy()
        ax.grid()
        n_vin = data.shape[-1]
        colors_map = ['#E4DACE', '#E5BB4B', '#498EAF', '#631F16']
        if len(data.shape) == 2:
            positions = np.arange(n_vin) + 1
            x_pos = None
            n_bag = 1
        else:
            n_bag = data.shape[-2]
            p = (np.linspace(0, 1, n_vin + 2) - 0.5) * bag_width
            positions = np.hstack([p[1:-1] + 0.5 + i for i in range(n_bag)]) * n_vin
            x_pos = np.arange(n_bag) * n_vin + n_vin / 2

        parts = ax.violinplot(data.reshape(data.shape[0], -1), widths=0.5 * bag_width, positions=positions,
                              showmeans=False, showmedians=False, showextrema=False)

        for i in range(n_vin):
            for j in range(n_bag):
                parts['bodies'][i + j * n_vin].set_facecolor(colors_map[i % len(colors_map)])  # violin color
                parts['bodies'][i + j * n_vin].set_edgecolor('grey')  # violin edge
                parts['bodies'][i + j * n_vin].set_alpha(0.9)
        ax.legend(legends)
        quartile1, medians, quartile3 = np.percentile(data.reshape(data.shape[0], -1), [25, 50, 75], axis=0)
        whiskers = np.array([
            adjacent_values(sorted_array, q1, q3)
            for sorted_array, q1, q3 in zip(data.reshape(data.shape[0], -1), quartile1, quartile3)])
        whiskers_min, whiskers_max = whiskers[:, 0], whiskers[:, 1]

        ax.scatter(positions, medians, marker='o', color='white', s=5, zorder=3)
        ax.vlines(positions, quartile1, quartile3, color='black', linestyle='-', lw=5)
        # ax.vlines(positions, whiskers_min, whiskers_max, color='black', linestyle='-', lw=1)
        if xticks is None:
            xticks = np.arange(n_vin * n_bag)
        ax.set_xlabel(xlabel)
        set_axis_style(ax, xticks, x_pos)

    def plot_fields_1D(self, fig, axs, real, pred, coord=None,
                       titles=None, xylabels=('x', 'field'), legends=None,
                       show_channel=None, field_names=None):
        r"""
            plot the fields of real and pred
            Args:
                :param fig:
                :param axs:
                :param real:
                :param pred:
                :param coord:
                :param titles:
                :param xylabels:
                :param legends:
                :param show_channel:
            :return:
                None
        """
        real = tensor2numpy(real)
        pred = tensor2numpy(pred)
        coord = tensor2numpy(coord)

        if len(axs.shape) == 1:
            axs = axs[None, :]

        if show_channel is None:
            show_channel = np.arange(real.shape[-1])

        if legends is None:
            if pred is not None:
                legends = ['true', 'pred', 'error']
            else:
                legends = ['true', ]

        if field_names is None:
            field_names = []
            for i in show_channel:
                field_names.append('field ' + str(i + 1))

        if coord is None:
            coord = np.arange(real.shape[0])

        for i in range(len(show_channel)):

            fi = show_channel[i]

            axs[i][0].cla()
            if pred is None:
                ff = [real[..., fi], ]
                axs[i][0].plot(coord, ff[0], color='steelblue', linewidth=3, label=legends[0])
            else:
                ff = [real[..., fi], pred[..., fi], real[..., fi] - pred[..., fi]]
                axs[i][0].plot(coord, ff[0], color='steelblue', linewidth=3, label=legends[0])
                axs[i][0].plot(coord, ff[1], '*', color='firebrick', linewidth=10, label=legends[1])
                axs[i][1].plot(coord, ff[2], color='forestgreen', linewidth=2, label=legends[2])

            for j in range(len(titles)):
                axs[i][j].legend(loc="best")
                axs[i][j].set_xlabel(xylabels[0])
                axs[i][j].set_ylabel(xylabels[1])
                axs[i][j].tick_params('both')
                axs[i][j].set_title(titles[j])

    def plot_fields_2D(self, fig, axs,
                       real, pred=None, coord=None,
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

        real = tensor2numpy(real)
        pred = tensor2numpy(pred)
        coord = tensor2numpy(coord)
        edges = tensor2numpy(edges)
        mask = tensor2numpy(mask)

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
            field_names = []
            for i in show_channel:
                field_names.append('field ' + str(i + 1))

        if fmin_max is None:
            faxis = tuple(range(len(real.shape) - 1))
            fmin, fmax = real.min(axis=faxis), real.max(axis=faxis)
        else:
            fmin, fmax = fmin_max[0], fmin_max[1]

        if cmin_max is None:
            caxis = tuple(range(len(coord.shape) - 1))
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

    def output_tecplot(self,
                       file,
                       real,
                       pred,
                       coord,
                       edges=None,
                       field_names=None,
                       zone_name='Zone'):

        real = tensor2numpy(real)
        pred = tensor2numpy(pred)
        coord = tensor2numpy(coord)
        edges = tensor2numpy(edges)

        name_real = ['True_' + name for name in field_names]
        name_pred = ['Pred_' + name for name in field_names]
        name_err = ['Err_' + name for name in field_names]
        spatial_dim = coord.shape[-1]

        if pred is None:
            output = np.concatenate((coord, real), axis=-1)
        else:
            output = np.concatenate((coord, real, pred, pred - real), axis=-1)

        df = pd.DataFrame(output.reshape(-1, output.shape[-1]))

        f = open(file, "w")
        f.write("%s\n" % ('TITLE = "Element Data"'))
        if spatial_dim == 1:
            f.write("%s" % ('VARIABLES = "X",'))
        elif spatial_dim == 2:
            f.write("%s" % ('VARIABLES = "X","Y",'))
        else:
            f.write("%s" % ('VARIABLES = "X","Y","Z",'))

        for i in range(len(name_real)):
            f.write("%s" % ('"' + name_real[i] + '",'))

        if pred is not None:
            for i in range(len(name_pred)):
                f.write("%s" % ('"' + name_pred[i] + '",'))
            for i in range(len(name_err)):
                f.write("%s" % ('"' + name_err[i] + '",'))

        if edges is None:
            f.write("\n%s" % ('ZONE T="' + zone_name + '", '))
            if len(coord.shape) == 2:
                f.write("%s" % ('I=' + str(coord.shape[0])))
            elif len(coord.shape) == 3:
                f.write("%s" % ('I=' + str(coord.shape[1]) + ', J=' + str(coord.shape[0])))
            else:
                f.write(
                    "%s" % ('I=' + str(coord.shape[0]) + ', J=' + str(coord.shape[1]) + ', K=' + str(coord.shape[2])))
            f.write("%s\n" % (', F=POINT'))
            f.close()
        else:
            # todo: add the unstructured mesh
            return NotImplementedError("The unstructured mesh is not implemented now!")

        df.to_csv(file, index=False, mode='a', float_format="%20.5e", sep=",", header=False)


if __name__ == '__main__':
    visual_model = Visual(use_tex='ch-en')

    # random data
    x = np.random.rand(10, 1)
    y = np.random.rand(10, 1)
    coord = np.concatenate([x, y], axis=-1)
    real = np.random.rand(10, 1)
    pred = np.random.rand(10, 1)
    mask = np.array([[0, 0], [0, 0.5], [0.5, 0.5], [0.5, 0]])
    fig, axs = plt.subplots(1, 3)
    visual_model.plot_fields_2D(fig, axs, real, pred, coord, mask=mask, titles=['真实field', '预测field', '误差field'])
    plt.show()  # fig.show();input();

    fig, axs = plt.subplots(1, 2, num=2)
    visual_model.plot_fields_1D(fig, axs, real, pred, titles=['预测field', '误差field'])
    plt.show()  # fig.show();input();

    # airfoil reading from msh & random fields
    # import meshio
    data_file = '../demo/cylinder_2d_t/data/ns_unsteady.npy'
    data = np.load(data_file, allow_pickle=True).item()
    u_ref = np.array(data["u"], dtype=np.float32)
    v_ref = np.array(data["v"], dtype=np.float32)
    p_ref = np.array(data["p"], dtype=np.float32)
    coords = np.array(data["coords"], dtype=np.float32)
    cylinder_coords = np.array(data["cylinder_coords"], dtype=np.float32)

    real = np.stack((p_ref[0], u_ref[0], v_ref[0]), axis=1)
    fig, axs = plt.subplots(3, 1)
    visual_model.plot_fields_2D(fig, axs, real, None, coords, mask=cylinder_coords,
                                titles=['真实field', '预测field', '误差field'], field_names=['p', 'u', 'v'])
    a = 1
    plt.show()  # fig.show();input();
