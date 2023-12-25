# _*_ coding: utf-8 _*_

"""
  @Time    : 2023.12.25 11:15
  @File    : model.py
  @Project : cnn_demo
  @Author  : Saitama
  @IDE     : PyCharm
  @Desc    : 
"""
from typing import Optional, Callable

import torch.nn as nn


def _make_divisible(ch, divisor=8, min_ch=None):
    """
    使传入的 channel 调整到最近的 8 的整数倍
    :param ch:
    :param divisor:
    :param min_ch:
    :return:
    """
    if min_ch is None:
        min_ch = divisor
    new_ch = max(min_ch, int(ch + divisor / 2) // divisor * divisor)
    if new_ch < 0.9 * ch:
        new_ch += divisor
    return new_ch


class ConvBnActivation(nn.Sequential):
    """
    卷积层 + BN层 + 激活函数
    """

    def __init__(self,
                 in_c: int,
                 out_c: int,
                 kernel_s: int = 3,
                 stride: int = 1,
                 groups: int = 1,
                 normalize_layer: Optional[Callable[..., nn.Module]] = None,
                 activation_layer: Optional[Callable[..., nn.Module]] = None):
        padding = (kernel_s - 1) // 2

        if normalize_layer is None:
            normalize_layer = nn.BatchNorm2d

        if activation_layer is None:
            activation_layer = nn.SiLU  # alias Swish

        super(ConvBnActivation, self).__init__(nn.Conv2d(in_channels=in_c,
                                                         out_channels=out_c,
                                                         kernel_size=kernel_s,
                                                         stride=stride,
                                                         padding=padding,
                                                         bias=False),
                                               normalize_layer(out_c),
                                               activation_layer())


class SqueezeExcitation(nn.Module):
    """
    SE 模块
    """

    def __init__(self,
                 in_c: int,  # block input channel
                 expand_c: int,  # block expand channel
                 squeeze_factor: int = 4
                 ):
        super(SqueezeExcitation, self).__init__()
        squeeze_c = in_c // squeeze_factor  # 第一个全连接层的输出维度，应为整个模块的输入矩阵的维度的 1/4
        self.fc1 = nn.Conv2d(expand_c, squeeze_c, kernel_size=1)  # 使用 1x1 的卷积替代全连接，可能是底层逻辑对卷积做了优化，可直接理解成全连接
        self.ac1 = nn.SiLU()  # 激活函数，alias SWISH
        self.fc2 = nn.Conv2d(squeeze_c, expand_c, kernel_size=1)
        self.ac2 = nn.Sigmoid()  # 激活函数，Sigmoid

    def forward(self, x):
        """
        正向传播过程
        :param x:
        :return:
        """
