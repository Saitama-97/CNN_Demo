# _*_ coding: utf-8 _*_

"""
  @Time    : 2023.12.25 11:15
  @File    : model.py
  @Project : cnn_demo
  @Author  : Saitama
  @IDE     : PyCharm
  @Desc    : 
"""
import copy
import math
from collections import OrderedDict
from functools import partial
from typing import Optional, Callable

import torch
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


# *********************** 论文中使用的 dropout 方法 ***********************
def drop_path(x, drop_prob: float = 0., training: bool = False):
    """
    Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).
    "Deep Networks with Stochastic Depth", https://arxiv.org/pdf/1603.09382.pdf

    This function is taken from the rwightman.
    It can be seen here:
    https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/layers/drop.py#L140
    """
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
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
                                                         groups=groups,
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
        【SE模块】正向传播过程
        :param x:
        :return:
        """
        scale = F.adaptive_avg_pool2d(x, output_size=(1, 1))
        scale = self.fc1(scale)
        self.ac1(scale)
        scale = self.fc2(scale)
        self.ac2(scale)

        return scale * x


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


class MBConv(nn.Module):
    """
    MBConv 模块
    """
    def __init__(self,
                 cnf: InvertedResidualConfig,
                 norm_layer: Callable[..., nn.Module]):

        super(MBConv, self).__init__()

        if cnf.stride not in [1, 2]:
            raise ValueError("illegal stride value")

        # 是否使用 shortcut【步距为1且输入channel等于输出channel时才使用shortcut】
        self.has_shortcut = (cnf.stride == 1 and cnf.input_c == cnf.out_c)

        # 定义一个有序字典，按顺序存放不同层
        layers = OrderedDict()
        activation_layer = nn.SiLU  # alias SWISH

        # layer-1：当n != 1时，使用 1x1 卷积层升维
        if cnf.input_c != cnf.expanded_c:
            layers.update({"expand_conv":
                               ConvBnActivation(in_c=cnf.input_c,
                                                out_c=cnf.expanded_c,
                                                kernel_s=1,
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
                                                   expand_c=cnf.expanded_c)})

        # layer-4，卷积BN无激活
        layers.update({"project_conv": ConvBnActivation(in_c=cnf.expanded_c,
                                                        out_c=cnf.out_c,
                                                        kernel_s=1,
                                                        normalize_layer=norm_layer,
                                                        activation_layer=nn.Identity)})  # nn.Identity()：Do Nothing !!!

        # 把上述组件整合成一个 MBConv 模块
        self.block = nn.Sequential(layers)
        self.out_channels = cnf.out_c
        self.is_strided = cnf.stride > 1  # ?

        if self.has_shortcut and cnf.drop_rate > 0:
            self.dropout = DropPath(cnf.drop_rate)
        else:
            self.dropout = nn.Identity()

    def forward(self, x):
        """
        正向传播过程
        :param x:
        :return:
        """
        result = self.block(x)
        result = self.dropout(result)

        # 如果存在 shortcut，则将输入与dropout的输出相加
        if self.has_shortcut:
            result += x

        return result


class EfficientNet(nn.Module):
    def __init__(self,
                 width_coefficient: float,
                 depth_coefficient: float,
                 num_classes: int = 1000,
                 dropout_rate: float = 0.2,
                 drop_connect_rate: float = 0.2,
                 block: Optional[Callable[..., nn.Module]] = None,
                 norm_layer: Optional[Callable[..., nn.Module]] = None):
        super(EfficientNet, self).__init__()

        # kernel_size, in_channel, out_channel, exp_ratio, stride, use_SE, drop_connect_rate, repeats
        default_cnf = [[3, 32, 16, 1, 1, True, drop_connect_rate, 1],
                       [3, 16, 24, 6, 2, True, drop_connect_rate, 2],
                       [5, 24, 40, 6, 2, True, drop_connect_rate, 2],
                       [3, 40, 80, 6, 2, True, drop_connect_rate, 3],
                       [5, 80, 112, 6, 1, True, drop_connect_rate, 3],
                       [5, 112, 192, 6, 2, True, drop_connect_rate, 4],
                       [3, 192, 320, 6, 1, True, drop_connect_rate, 1]]

        def round_repeats(repeats):
            """
            根据深度扩展因子，调整每个 stage 的深度（每个 stage 中 block 的重复次数）
            :param repeats:
            :return:
            """
            return int(math.ceil(depth_coefficient * repeats))

        if block is None:
            block = MBConv

        if norm_layer is None:
            norm_layer = partial(nn.BatchNorm2d, eps=1e-3, momentum=0.1)

        adjust_channels = partial(InvertedResidualConfig.adjust_channels, width_coefficient=width_coefficient)

        bneck_conf = partial(InvertedResidualConfig, width_coefficient=width_coefficient)

        # MBConv 模块总数
        num_blocks = float(sum(round_repeats(i[-1]) for i in default_cnf))

        # 记录每个MBConv模块的参数配置信息
        inverted_residual_setting = []
        b = 0
        for stage, args in enumerate(default_cnf):
            cnf = copy.copy(args)
            # 每个stage中block个数
            for i in range(round_repeats(cnf.pop(-1))):
                if i > 0:
                    cnf[-3] = 1  # 非第一个block，stride为1
                    cnf[1] = cnf[2]  # 非第一个block，输入输出channel相等
                cnf[-1] = args[-2] * b / num_blocks  # 更新dropout ratio
                index = str(stage + 1) + chr(i + 97)  # 第几个stage的第几个block
                inverted_residual_setting.append(bneck_conf(*cnf, index))  # 生成当前stage的参数配置，存放到配置列表中
                b += 1

        # 网络层级结构
        layers = OrderedDict()

        # 第一层：3x3卷积层【Stage-1】
        layers.update({"stem_conv": ConvBnActivation(in_c=3,
                                                     out_c=adjust_channels(32),
                                                     kernel_s=3,
                                                     stride=2,
                                                     normalize_layer=norm_layer)})
        # 第二~八层：MBConv模块层【Stage2~8】
        for cnf in inverted_residual_setting:
            layers.update({cnf.index: block(cnf, norm_layer)})

        # 第九层：1x1卷积层，输入channel等于Stage-8的输出channel
        last_conv_input_c = inverted_residual_setting[-1].out_c
        last_conv_output_c = adjust_channels(1280)
        layers.update({"top": ConvBnActivation(in_c=last_conv_input_c,
                                               out_c=last_conv_output_c,
                                               kernel_s=1,
                                               stride=1,
                                               normalize_layer=norm_layer)})

        # 第九层：池化层
        self.features = nn.Sequential(layers)
        self.avgpool = nn.AdaptiveAvgPool2d(1)

        # 第九层：全连接层
        classifier = []
        if dropout_rate > 0:
            # 如果dropout_rate不为0，需要先进行dropout，才能进入全连接层
            classifier.append(nn.Dropout2d(p=dropout_rate, inplace=True))
        classifier.append(nn.Linear(in_features=last_conv_output_c, out_features=num_classes))
        self.classifier = nn.Sequential(*classifier)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        """
        正向传播过程
        :param x:
        :return:
        """
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)

        return x


def generate_efficientnet_b0(num_classes=1000):
    return EfficientNet(width_coefficient=1.0,
                        depth_coefficient=1.0,
                        num_classes=num_classes,
                        dropout_rate=0.2,
                        drop_connect_rate=0.2)


def generate_efficientnet_b1(num_classes=1000):
    return EfficientNet(width_coefficient=1.0,
                        depth_coefficient=1.1,
                        num_classes=num_classes,
                        dropout_rate=0.2,
                        drop_connect_rate=0.2)


def generate_efficientnet_b2(num_classes=1000):
    return EfficientNet(width_coefficient=1.1,
                        depth_coefficient=1.2,
                        num_classes=num_classes,
                        dropout_rate=0.3,
                        drop_connect_rate=0.2)


def generate_efficientnet_b3(num_classes=1000):
    return EfficientNet(width_coefficient=1.2,
                        depth_coefficient=1.4,
                        num_classes=num_classes,
                        dropout_rate=0.3,
                        drop_connect_rate=0.2)


def generate_efficientnet_b4(num_classes=1000):
    return EfficientNet(width_coefficient=1.4,
                        depth_coefficient=1.8,
                        num_classes=num_classes,
                        dropout_rate=0.4,
                        drop_connect_rate=0.2)


def generate_efficientnet_b5(num_classes=1000):
    return EfficientNet(width_coefficient=1.6,
                        depth_coefficient=2.2,
                        num_classes=num_classes,
                        dropout_rate=0.4,
                        drop_connect_rate=0.2)


def generate_efficientnet_b6(num_classes=1000):
    return EfficientNet(width_coefficient=1.8,
                        depth_coefficient=2.6,
                        num_classes=num_classes,
                        dropout_rate=0.5,
                        drop_connect_rate=0.2)


def generate_efficientnet_b7(num_classes=1000):
    return EfficientNet(width_coefficient=2.0,
                        depth_coefficient=3.1,
                        num_classes=num_classes,
                        dropout_rate=0.5,
                        drop_connect_rate=0.2)

#
