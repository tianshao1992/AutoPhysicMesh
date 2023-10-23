#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
# @Copyright (c) 2022 Baidu.com, Inc. All Rights Reserved
# @Time    : 2022/11/6 17:37
# @Author  : Liu Tianyuan (liutianyuan02@baidu.com)
# @Site    :
# @File    : spectral_layers.py
"""

# add configs.py path
# file_path = os.path.abspath(os.path.dirname(__file__))
# sys.path.append(os.path.join(file_path.split('fno')[0]))
# sys.path.append(os.path.join(file_path.split('Models')[0]))

from Module import bkd, nn
from Module.activations import get as get_activation
from NetZoo.fno.basic_operators import complex_mul1d, complex_mul2d, complex_mul3d

class SpectralConv1d(nn.Module):
    '''
    1维谱卷积
    Modified Zongyi Li's Spectral1dConv code
    https://github.com/zongyi-li/fourier_neural_operator/blob/master/fourier_1d.py
    '''

    def __init__(self,
                 input_dim: int,
                 output_dim: int,
                 spectral_modes: tuple or list,  # number of fourier modes
                 spectral_norm: str = "ortho",
                 layer_active: str = 'gelu',
                 layer_dropout: float = 0.1,
                 return_freq: bool = False,
                 use_complex: bool = True,
                 ):
        super(SpectralConv1d, self).__init__()

        """
        1D Fourier layer. It does FFT, linear transform, and Inverse FFT.    
        """

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.spectral_modes = spectral_modes  # Number of Fourier modes to multiply, at most floor(N/2) + 1
        self.spectral_norm = spectral_norm

        self.return_freq = return_freq
        self.use_complex = use_complex

        self.layer_active = get_activation(layer_active)
        self.layer_dropout = nn.Dropout(layer_dropout)
        self.layer_linear = nn.Conv1d(self.input_dim, self.output_dim, kernel_size=1)  # for residual
        # self.linear = nn.Linear(self.in_dim, self.out_dim)
        self.scale = (1 / (input_dim * output_dim))

        if self.use_complex:
            self.weights = nn.Parameter(self.scale * bkd.rand(input_dim, output_dim, self.spectral_modes[0],
                                                              dtype=bkd.cfloat))
        else:
            self.weights = nn.Parameter(self.scale * bkd.rand(input_dim, output_dim, self.spectral_modes[0], 2,
                                                              dtype=bkd.float32))
        # xavier_normal_(self.weights1, gain=1 / (in_dim * out_dim))

    def forward(self, x):
        """
        forward computation
        """
        batchsize = x.shape[0]
        # Compute Fourier coeffcients up to factor of e^(- something constant)
        res = self.layer_linear(x)
        # x = self.dropout(x)

        # Multiply relevant Fourier modes
        if self.use_complex:
            x_ft = bkd.fft.rfft(x, norm=self.spectral_norm)
            out_ft = bkd.zeros(batchsize, self.output_dim, x.size(-1) // 2 + 1, device=x.device, dtype=bkd.cfloat)

        else:
            x_ft = bkd.view_as_real(bkd.fft.rfft(x, norm=self.spectral_norm))
            out_ft = bkd.zeros(batchsize, self.output_dim, x.size(-1) // 2 + 1, 2,
                               device=x.device, dtype=bkd.float32)

        out_ft[:, :, :self.spectral_modes[0]] = complex_mul1d(x_ft[:, :, :self.spectral_modes[0]],
                                                           self.weights, self.use_complex)

        # Return to physical space
        if self.use_complex:
            x = bkd.fft.irfft(out_ft, norm=self.spectral_norm)
        else:
            x = bkd.fft.irfft(bkd.view_as_complex(out_ft), norm=self.spectral_norm)

        x = self.layer_active(x + res)

        if self.return_freq:
            return x, out_ft
        else:
            return x


class SpectralConv2d(nn.Module):
    '''
    2维谱卷积
    Modified Zongyi Li's SpectralConv2d PyTorch 1.6 code
    using only real weights
    https://github.com/zongyi-li/fourier_neural_operator/blob/master/fourier_2d.py
    '''

    def __init__(self,
                 input_dim: int,
                 output_dim: int,
                 spectral_modes: list or tuple,  # number of fourier modes
                 spectral_norm: str = "ortho",
                 layer_active: str or callable = 'gelu',
                 layer_dropout: float = 0.1,
                 return_freq: bool = False,
                 use_complex: bool = True,
                 ):
        super(SpectralConv2d, self).__init__()

        """
        2D Fourier layer. It does FFT, linear transform, and Inverse FFT.    
        """

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.spectral_modes = spectral_modes  # Number of Fourier modes to multiply, at most floor(N/2) + 1

        self.spectral_norm = spectral_norm
        self.layer_dropout = nn.Dropout(layer_dropout)
        self.layer_active = get_activation(layer_active)
        self.return_freq = return_freq
        self.use_complex = use_complex
        self.layer_linear = nn.Conv2d(self.input_dim, self.output_dim, kernel_size=1)  # for residual

        self.scale = (1 / (input_dim * output_dim))

        self.weights = nn.ParameterList()
        if self.use_complex:
            for i in range(2):
                self.weights.append(nn.Parameter(
                    self.scale * bkd.rand(input_dim, output_dim,
                                          self.spectral_modes[0], self.spectral_modes[1], dtype=bkd.cfloat)))


        else:

            for i in range(2):
                self.weights.append(nn.Parameter(
                    self.scale * bkd.rand(input_dim, output_dim,
                                          self.spectral_modes[0], self.spectral_modes[1], 2, dtype=bkd.float32)))


    # Complex multiplication

    def forward(self, x):
        """
        forward computation
        """
        batch_size = x.shape[0]
        # Compute Fourier coeffcients up to factor of e^(- something constant)
        res = self.layer_linear(x)
        x = self.layer_dropout(x)
        x_ft = bkd.fft.rfft2(x, norm=self.spectral_norm)

        # Multiply relevant Fourier modes
        out_ft = bkd.zeros(batch_size, self.output_dim, x.size(-2), x.size(-1) // 2 + 1,
                           dtype=bkd.cfloat, device=x.device)
        out_ft[:, :, :self.spectral_modes[0], :self.spectral_modes[1]] = \
            complex_mul2d(x_ft[:, :, :self.spectral_modes[0], :self.spectral_modes[1]],
                          self.weights[0], use_complex=self.use_complex)
        out_ft[:, :, -self.spectral_modes[0]:, :self.spectral_modes[1]] = \
            complex_mul2d(x_ft[:, :, -self.spectral_modes[0]:, :self.spectral_modes[1]],
                          self.weights[1], use_complex=self.use_complex)

        # Return to physical space
        x = bkd.fft.irfft2(out_ft, s=(x.size(-2), x.size(-1)), norm=self.spectral_norm)
        x = self.layer_active(x + res)

        if self.return_freq:
            return x, out_ft
        else:
            return x


class SpectralConv3d(nn.Module):
    '''
    三维谱卷积
    Modified Zongyi Li's SpectralConv2d PyTorch 1.6 code
    using only real weights
    https://github.com/zongyi-li/fourier_neural_operator/blob/master/fourier_2d.py
    '''

    def __init__(self,
                 input_dim: int,
                 output_dim: int,
                 spectral_modes: list or tuple,  # number of fourier modes
                 spectral_norm: str = "ortho",
                 layer_active: str or callable = 'gelu',
                 layer_dropout: float = 0.1,
                 return_freq: bool = False,
                 use_complex: bool = True,
                 ):  # whether to return the frequency target
        super(SpectralConv3d, self).__init__()

        """
        3D Fourier layer. It does FFT, linear transform, and Inverse FFT.    
        """

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.spectral_modes = spectral_modes  # Number of Fourier modes to multiply, at most floor(N/2) + 1
        self.spectral_norm = spectral_norm

        self.layer_dropout = nn.Dropout(layer_dropout)
        self.layer_active = get_activation(layer_active)
        self.return_freq = return_freq
        self.use_complex = use_complex
        self.layer_linear = nn.Conv3d(self.input_dim, self.output_dim, kernel_size=1)  # for residual
        self.scale = (1 / (input_dim * output_dim))

        self.weights = nn.ParameterList()
        if self.use_complex:
            for i in range(4):
                self.weights.append(
                    nn.Parameter(self.scale * bkd.rand(input_dim, output_dim,
                                 self.spectral_modes[0],
                                 self.spectral_modes[1],
                                 self.spectral_modes[2],
                                 dtype=bkd.cfloat)))

        else:
            for i in range(4):
                self.weights.append(
                    nn.Parameter(self.scale * bkd.rand(input_dim, output_dim,
                                 self.spectral_modes[0],
                                 self.spectral_modes[1],
                                 self.spectral_modes[2],
                                 2,
                                 dtype=bkd.float32)))

    # Complex multiplication

    def forward(self, x):
        """
        forward computation
        """
        batch_size = x.size(0)
        # Compute Fourier coeffcients up to factor of e^(- something constant)
        res = self.layer_linear(x)
        # x = self.dropout(x)

        # Multiply relevant Fourier modes
        if self.use_complex:
            x_ft = bkd.fft.rfftn(x, dim=[-3, -2, -1], norm=self.spectral_norm)
            out_ft = bkd.zeros(batch_size, self.output_dim, x.size(-3), x.size(-2), x.size(-1) // 2 + 1,
                               dtype=bkd.cfloat, device=x.device)
        else:
            x_ft = bkd.view_as_real(bkd.fft.rfftn(x, dim=[-3, -2, -1], norm=self.spectral_norm))
            out_ft = bkd.zeros(batch_size, self.output_dim, x.size(-3), x.size(-2), x.size(-1) // 2 + 1, 2,
                               dtype=bkd.float32, device=x.device)

        out_ft[:, :, :self.spectral_modes[0], :self.spectral_modes[1], :self.spectral_modes[2]] = \
            complex_mul3d(x_ft[:, :, :self.spectral_modes[0], :self.spectral_modes[1], :self.spectral_modes[2]],
                          self.weights[0], use_complex=self.use_complex)
        out_ft[:, :, -self.spectral_modes[0]:, :self.spectral_modes[1], :self.spectral_modes[2]] = \
            complex_mul3d(x_ft[:, :, -self.spectral_modes[0]:, :self.spectral_modes[1], :self.spectral_modes[2]],
                          self.weights[1], use_complex=self.use_complex)
        out_ft[:, :, :self.spectral_modes[0], -self.spectral_modes[1]:, :self.spectral_modes[2]] = \
            complex_mul3d(x_ft[:, :, :self.spectral_modes[0], -self.spectral_modes[1]:, :self.spectral_modes[2]],
                          self.weights[2], use_complex=self.use_complex)
        out_ft[:, :, -self.spectral_modes[0]:, -self.spectral_modes[1]:, :self.spectral_modes[2]] = \
            complex_mul3d(x_ft[:, :, -self.spectral_modes[0]:, -self.spectral_modes[1]:, :self.spectral_modes[2]],
                          self.weights[3], use_complex=self.use_complex)

        # Return to physical space
        if self.use_complex:
            x = bkd.fft.irfftn(out_ft, s=(x.size(-3), x.size(-2), x.size(-1)), norm=self.spectral_norm)
        else:
            x = bkd.fft.irfftn(bkd.view_as_complex(out_ft), s=(x.size(-3), x.size(-2), x.size(-1)),
                               norm=self.spectral_norm)

        x = self.layer_active(x + res)

        if self.return_freq:
            return x, out_ft
        else:
            return x

if __name__ == '__main__':
    x = bkd.ones([10, 3, 64])
    layer = SpectralConv1d(input_dim=3, output_dim=10, spectral_modes=[5,], use_complex=False)
    y = layer(x)
    print(y.shape)

    lossfunc = bkd.nn.MSELoss()
    optimizer = bkd.optim.Adam(layer.parameters(), lr=0.001)
    loss = lossfunc(y, bkd.ones_like(y))
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    #
    x = bkd.ones([10, 3, 55, 64])
    layer = SpectralConv2d(input_dim=3, output_dim=10, spectral_modes=(5, 3))
    y = layer(x)
    print(y.shape)
    #
    x = bkd.ones([10, 3, 16, 32, 48])
    layer = SpectralConv3d(input_dim=3, output_dim=4, spectral_modes=(5, 5, 5), use_complex=True)
    y = layer(x)
    print(y.shape)
    #
    # x = torch.ones(10, 64, 128)
    # layer = AdaptiveFourier1d(hidden_size=64, num_blocks=4)
    # y = layer(x)
    # print(y.shape)
    #
    # x = torch.ones(10, 64, 55, 64)
    # layer = AdaptiveFourier2d(hidden_size=64, num_blocks=4)
    # y = layer(x)
    # print(y.shape)
    #
    # x = torch.ones(10, 64, 55, 64, 33)
    # layer = AdaptiveFourier3d(hidden_size=64, num_blocks=4)
    # y = layer(x)
    # print(y.shape)
