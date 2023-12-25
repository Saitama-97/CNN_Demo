# _*_ coding: utf-8 _*_

"""
  @Time    : 2023.12.25 11:15
  @File    : model.py
  @Project : cnn_demo
  @Author  : Saitama
  @IDE     : PyCharm
  @Desc    : 
"""
from collections import OrderedDict
from typing import Optional, Callable

import torch.nn as nn
import torch.nn.functional as F


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
        scale = F.adaptive_avg_pool2d(x, output_size=(1, 1))
        scale = self.fc1(scale)
        self.ac1(scale)
        scale = self.fc2(scale)
        self.ac2(scale)

        return x * scale


class InvertedResidualConfig:
    def __init__(self,
                 kernel: int,
                 input_c: int,
                 out_c: int,
                 expand_ratio: int,
                 stride: int,
                 use_se: bool,
                 drop_rate: float,
                 index: str,
                 width_coefficient: float):
        self.kernel = kernel
        self.input_c = self.adjust_channels(input_c, width_coefficient)
        self.expanded_c = self.input_c * expand_ratio
        self.out_c = self.adjust_channels(out_c, width_coefficient)
        self.use_se = use_se
        self.stride = stride
        self.drop_rate = drop_rate
        self.index = index

    @staticmethod
    def adjust_channels(channels: int, width_coefficient: float):
        return _make_divisible(channels * width_coefficient, 8)


class InvertedResidual(nn.Module):
    def __init__(self,
                 cnf: InvertedResidualConfig,
                 norm_layer: Callable[..., nn.Module]):
        super(InvertedResidual, self).__init__()

        if cnf.stride not in [1, 2]:
            raise ValueError("illegal stride value")

        # 是否使用 shortcut
        self.use_connect = (cnf.stride == 1) and (cnf.input_c == cnf.out_c)

        # 定义一个有序字典，按顺序存放不同层
        layers = OrderedDict()
        activation_layer = nn.SiLU  # alias SWISH

        # layer-1：当n != 1时，使用 1x1 卷积层升维
        if cnf.input_c != cnf.expanded_c:
            layers.update({"expand_conv":
                               ConvBnActivation(in_c=cnf.input_c,
                                                out_c=cnf.expanded_c,
                                                kernel_s=1,
                                                stride=1,
                                                groups=1,
                                                normalize_layer=norm_layer,
                                                activation_layer=activation_layer)})

        # layer-2，DW 卷积层
        layers.update({"dwconv": ConvBnActivation(in_c=cnf.expanded_c,
                                                  out_c=cnf.expanded_c,
                                                  kernel_s=cnf.kernel,
                                                  stride=cnf.stride,
                                                  groups=cnf.expanded_c,
                                                  normalize_layer=norm_layer,
                                                  activation_layer=activation_layer)})

        # layer-3，SE模块
        if cnf.use_se:
            layers.update({"se": SqueezeExcitation(in_c=cnf.input_c,
                                                   expand_c=cnf.expanded_c,
                                                   squeeze_factor=4)})

        # layer-4，卷积BN无激活
        layers.update({"project_conv": ConvBnActivation(in_c=cnf.expanded_c,
                                                        out_c=cnf.out_c,
                                                        kernel_s=1,
                                                        stride=1,
                                                        groups=1,
                                                        normalize_layer=norm_layer,
                                                        activation_layer=nn.Identity)})  # nn.Identity()：Do Nothing !!!

        # 把上述组件整合成一个 MBConv 模块
        self.block = nn.Sequential(layers)
        self.out_channels = cnf.out_c
        self.is_strided = cnf.stride > 1  # ?

        if cnf.drop_rate > 0:
            self.dropout = nn.Dropout2d(p=cnf.drop_rate, inplace=True)
        else:
            self.dropout = nn.Identity()

    def forward(self, x):
        """
        正向传播过程
        :param x:
        :return:
        """
        result = self.block(x)
        result = self.dropout(x)

        # 如果存在 shortcut，则将输入与dropout的输出相加
        if self.use_connect:
            result += x

        return result
