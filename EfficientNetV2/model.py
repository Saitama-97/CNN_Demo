# _*_ coding: utf*8 _*_

"""
  @Time    : 2023.12.26 16:55
  @File    : model.py
  @Project : CNN_Demo
  @Author  : Saitama
  @IDE     : PyCharm
  @Desc    : 手搓 EfficientNet*V2
            网络定义
"""
from collections import OrderedDict
from typing import Optional, Callable

import torch
import torch.nn as nn


# *********************** 论文中使用的 dropout 方法 ***********************
def drop_path(x, drop_prob: float = 0., training: bool = False):
    """
    Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).
    "Deep Networks with Stochastic Depth", https://arxiv.org/pdf/1603.09382.pdf

    This function is taken from the rwightman.
    It can be seen here:
    https://github.com/rwightman/pytorch*image*models/blob/master/timm/models/layers/drop.py#L140
    """
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 * drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim * 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()  # binarize
    output = x.div(keep_prob) * random_tensor
    return output


class DropPath(nn.Module):
    """
    Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    "Deep Networks with Stochastic Depth", https://arxiv.org/pdf/1603.09382.pdf
    """

    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)


# *********************** 论文中使用的 dropout 方法 ***********************

class ConvBnAct(nn.Module):
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
        super(ConvBnAct, self).__init__()

        padding = (kernel_s - 1) // 2

        if normalize_layer is None:
            normalize_layer = nn.BatchNorm2d

        if activation_layer is None:
            activation_layer = nn.SiLU  # alias Swish

        # 卷积层
        self.conv = nn.Conv2d(in_channels=in_c,
                              out_channels=out_c,
                              kernel_size=kernel_s,
                              stride=stride,
                              padding=padding,
                              groups=groups,
                              bias=False)
        # BN层（输入channel等于卷积层的输出channel）
        self.normalize_layer = normalize_layer(out_c)
        # 激活函数
        self.activation_layer = activation_layer()

    def forward(self, x):
        """
        【卷积层 + BN层 + 激活函数】的正向传播过程
        :param x:
        :return:
        """
        result = self.conv(x)
        result = self.normalize_layer(result)
        result = self.activation_layer(result)

        return result


class SqueezeExcitation(nn.Module):
    """
    SE 模块
    """

    def __init__(self,
                 in_c: int,  # block input channel
                 expand_c: int,  # block expand channel
                 squeeze_ratio: float = 0.25
                 ):
        super(SqueezeExcitation, self).__init__()
        squeeze_c = int(in_c * squeeze_ratio)  # 第一个全连接层的输出维度，应为整个模块的输入矩阵的维度的 1/4
        self.fc1 = nn.Conv2d(expand_c, squeeze_c, kernel_size=1)  # 使用 1x1 的卷积替代全连接，可能是底层逻辑对卷积做了优化，可直接理解成全连接
        self.ac1 = nn.SiLU()  # 激活函数，alias SWISH
        self.fc2 = nn.Conv2d(squeeze_c, expand_c, kernel_size=1)
        self.ac2 = nn.Sigmoid()  # 激活函数，Sigmoid

    def forward(self, x):
        """
        【SE模块】正向传播过程
        :param x:
        :return:
        """
        scale = x.mean((2, 3), keepdim=True)  # AvgPool，平均池化
        scale = self.fc1(scale)
        self.ac1(scale)
        scale = self.fc2(scale)
        self.ac2(scale)

        return scale * x


class MBConv(nn.Module):
    """
    MBConv 模块
    """

    def __init__(self,
                 kernel_s: int,
                 input_c: int,
                 output_c: int,
                 expand_ratio: int,
                 stride: int,
                 se_ratio: float,
                 drop_rate: float,
                 norm_layer: Callable[..., nn.Module]):
        super(MBConv, self).__init__()

        if stride not in [1, 2]:
            raise ValueError("illegal stride value")

        # 是否使用 shortcut【步距为1且输入channel等于输出channel时才使用shortcut】
        self.has_shortcut = (stride == 1 and in_c == out_c)

        # 激活函数
        activation_layer = nn.SiLU  # alias SWISH

        # layer-1: 1x1 卷积层，用于升维【在V1中可能因为expand_ratio=1不存在，但是V2必存在】
        # 1x1 卷积层升维后的channel【输入特征的维度 in_c * 扩展率 expand_ratio】
        expand_c = int(input_c * expand_ratio)
        self.expand_conv = ConvBnAct(in_c=input_c,
                                     out_c=output_c,
                                     kernel_s=kernel_s,
                                     stride=stride,
                                     normalize_layer=norm_layer,
                                     activation_layer=activation_layer)

        # layer-2：DW 卷积层【输入输出维度相等】
        self.dwconv = ConvBnAct(
            in_c=output_c,
            out_c=output_c,
            kernel_s=3,
            stride=stride,
            groups=output_c,  # DW 卷积的卷积核个数与输入维度相等
            normalize_layer=norm_layer,
            activation_layer=activation_layer
        )

        # layer-3：SE 模块【因为V2中SE_ratio恒大于0，所以一定使用SE模块】
        self.se = SqueezeExcitation(in_c=input_c,
                                    expand_c=expand_c,
                                    squeeze_factor=se_ratio)
