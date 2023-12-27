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
from functools import partial
from typing import Optional, Callable

import torch
import torch.nn as nn
from torch import Tensor


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
        self.bn = normalize_layer(out_c)
        # 激活函数
        self.act = activation_layer()

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
        self.conv_reduce = nn.Conv2d(expand_c, squeeze_c, kernel_size=1)  # 使用 1x1 的卷积替代全连接，可能是底层逻辑对卷积做了优化，可直接理解成全连接
        self.act1 = nn.SiLU()  # 激活函数，alias SWISH
        self.conv_expand = nn.Conv2d(squeeze_c, expand_c, kernel_size=1)
        self.act2 = nn.Sigmoid()  # 激活函数，Sigmoid

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
        self.has_shortcut = (stride == 1 and input_c == output_c)

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
            in_c=expand_c,
            out_c=expand_c,
            kernel_s=kernel_s,
            stride=stride,
            groups=output_c,  # DW 卷积的卷积核个数与输入维度相等
            normalize_layer=norm_layer,
            activation_layer=activation_layer
        )

        # layer-3：SE 模块【因为V2中SE_ratio恒大于0，所以一定使用SE模块】
        self.se = SqueezeExcitation(in_c=input_c,
                                    expand_c=expand_c,
                                    squeeze_ratio=se_ratio) if se_ratio > 0 else nn.Identity()

        # layer-4：1x1 卷积层+BN层，无激活函数（所以使用nn.Identity）
        self.project_conv = ConvBnAct(in_c=expand_c,
                                      out_c=output_c,
                                      kernel_s=1,
                                      normalize_layer=norm_layer,
                                      activation_layer=nn.Identity)  # 此处没有激活函数

        # 只有在使用shortcut连接时，才使用dropout层
        self.drop_rate = drop_rate
        if self.has_shortcut and drop_rate > 0:
            self.dropout = DropPath(drop_prob=drop_rate)

    def forward(self, x: Tensor) -> Tensor:
        """
        【MBConv模块】正向传播过程
        :param x:
        :return:
        """
        result = self.expand_conv(x)
        result = self.dwconv(result)
        result = self.se(result)
        result = self.project_conv(result)

        if self.has_shortcut:
            if self.drop_rate > 0:
                result = self.dropout(result)
            result += x

        return result


class FusedMBConv(nn.Module):
    """
    FusedMBConv 模块
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
        super(FusedMBConv, self).__init__()

        if stride not in [1, 2]:
            raise ValueError("illegal stride value")

        self.has_shortcut = (stride == 1 and input_c == output_c)

        # 激活函数
        activation_layer = nn.SiLU

        # expand_ratio不为1时，表示需要先进行升维
        self.has_expansion = (expand_ratio != 1)

        # 升维之后的维度【输入维度*扩展因子】
        expand_c = input_c * expand_ratio

        # expand_ratio不为1时，表示需要先进行升维
        if self.has_expansion:
            self.expand_conv = ConvBnAct(in_c=input_c,
                                         out_c=output_c,
                                         kernel_s=kernel_s,
                                         stride=stride,
                                         normalize_layer=norm_layer,
                                         activation_layer=activation_layer)

            self.project_conv = ConvBnAct(in_c=expand_c,
                                          out_c=output_c,
                                          kernel_s=1,
                                          normalize_layer=norm_layer,
                                          activation_layer=nn.Identity)  # 这边不需要激活函数
        else:  # expand_ratio不为1时，表示需要先进行升维
            self.project_conv = ConvBnAct(in_c=input_c,
                                          out_c=output_c,
                                          kernel_s=kernel_s,
                                          stride=stride,
                                          normalize_layer=norm_layer,
                                          activation_layer=activation_layer)

        # 只有在使用shortcut连接时，才使用dropout层
        self.drop_rate = drop_rate
        if self.has_shortcut and drop_rate > 0:
            self.dropout = DropPath(drop_prob=drop_rate)

    def forward(self, x: Tensor) -> Tensor:
        """
        【FusedMBConv模块】正向传播过程
        :param x:
        :return:
        """
        if self.has_expansion:
            result = self.expand_conv(x)
            result = self.project_conv(result)
        else:
            result = self.project_conv(x)

        if self.has_shortcut:
            if self.drop_rate > 0:
                result = self.dropout(result)
            result += x

        return result


class EfficientNetV2(nn.Module):
    """
    EfficientNetV2
    """

    def __init__(self,
                 model_cnf: list,  # [repeat, kernel, stride, expansion, in_c, out_c, operator, se_ratio]
                 num_classes: int = 1000,
                 num_features: int = 1280,
                 dropout_rate: float = 0.2,
                 drop_connect_rate: float = 0.2):
        super(EfficientNetV2, self).__init__()

        # 确保传入的模型配置格式正确
        # [repeat, kernel, stride, expansion, in_c, out_c, operator(default:0 for MBConv), se_ratio]
        for cnf in model_cnf:
            assert len(cnf) == 8

        norm_layer = partial(nn.BatchNorm2d, eps=1e-3, momentum=0.1)

        # 第一层：3x3卷积层【Stage-0】
        self.stem = ConvBnAct(in_c=3,
                              out_c=model_cnf[0][4],  # 对应配置第一层的输入维度 in_c
                              kernel_s=3,
                              stride=2,
                              normalize_layer=norm_layer)

        # 第二层，Block构成[MBConv+FusedMBConv]【Stage-1 ~ Stage-6】
        total_blocks = sum(i[0] for i in model_cnf)

        # 存放所有的Block
        blocks = list()

        block_id = 0

        for cnf in model_cnf:
            # 根据配置信息决定使用哪个模块
            module = FusedMBConv if cnf[-2] != 0 else MBConv
            for i in range(cnf[0]):  # 重复多少次
                blocks.append(module(kernel_s=cnf[1],
                                     input_c=cnf[4] if i == 0 else cnf[5],
                                     output_c=cnf[5],
                                     expand_ratio=cnf[3],
                                     stride=cnf[2] if i == 0 else 1,
                                     se_ratio=cnf[-1],
                                     drop_rate=drop_connect_rate * block_id / total_blocks,
                                     norm_layer=norm_layer))
                block_id += 1

        self.blocks = nn.Sequential(*blocks)

        # 第三层，1x1卷积层+池化层+全连接层【Stage-7】
        head = OrderedDict()
        head_input_c = model_cnf[-1][-3]  # 对应Stage-6的最后一层的输出维度

        # 1x1卷积层
        head.update({"project_conv": ConvBnAct(in_c=head_input_c,
                                               out_c=num_features,
                                               kernel_s=1,
                                               normalize_layer=norm_layer)})

        # 池化层
        head.update({"avgpool": nn.AdaptiveAvgPool2d(1)})

        # 展平处理
        head.update({"flatten": nn.Flatten()})

        # 如果有dropout层
        if dropout_rate > 0:
            head.update({"dropout": nn.Dropout(p=dropout_rate, inplace=True)})

        # 全连接层
        head.update({"classifier": nn.Linear(in_features=num_features, out_features=num_classes)})

        self.head = nn.Sequential(head)

        # 初始化权重
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

    def forward(self, x: Tensor) -> Tensor:
        """
        EfficientNet-V2 正向传播过程
        :param x:
        :return:
        """
        result = self.stem(x)
        result = self.blocks(result)
        result = self.head(result)

        return result


def generate_efficientnetv2_s(num_classes: int = 1000):
    """
    生成 EfficientNet-V2-S 网络
    train_size: 300, eval_size: 384
    :param num_classes:
    :return:
    """
    # repeat, kernel, stride, expansion, in_c, out_c, operator, se_ratio
    model_config = [[2, 3, 1, 1, 24, 24, 0, 0],
                    [4, 3, 2, 4, 24, 48, 0, 0],
                    [4, 3, 2, 4, 48, 64, 0, 0],
                    [6, 3, 2, 4, 64, 128, 1, 0.25],
                    [9, 3, 1, 6, 128, 160, 1, 0.25],
                    [15, 3, 2, 6, 160, 256, 1, 0.25]]

    return EfficientNetV2(model_cnf=model_config,
                          num_classes=num_classes,
                          dropout_rate=0.2)


def generate_efficientnetv2_m(num_classes: int = 1000):
    """
    生成 EfficientNet-V2-M 网络
    train_size: 384, eval_size: 480
    :param num_classes:
    :return:
    """

    # repeat, kernel, stride, expansion, in_c, out_c, operator, se_ratio
    model_config = [[3, 3, 1, 1, 24, 24, 0, 0],
                    [5, 3, 2, 4, 24, 48, 0, 0],
                    [5, 3, 2, 4, 48, 80, 0, 0],
                    [7, 3, 2, 4, 80, 160, 1, 0.25],
                    [14, 3, 1, 6, 160, 176, 1, 0.25],
                    [18, 3, 2, 6, 176, 304, 1, 0.25],
                    [5, 3, 1, 6, 304, 512, 1, 0.25]]

    return EfficientNetV2(model_cnf=model_config,
                          num_classes=num_classes,
                          dropout_rate=0.3)


def generate_efficientnetv2_l(num_classes: int = 1000):
    """
    生成 EfficientNet-V2-L 网络
    train_size: 384, eval_size: 480
    :param num_classes:
    :return:
    """
    # repeat, kernel, stride, expansion, in_c, out_c, operator, se_ratio
    model_config = [[4, 3, 1, 1, 32, 32, 0, 0],
                    [7, 3, 2, 4, 32, 64, 0, 0],
                    [7, 3, 2, 4, 64, 96, 0, 0],
                    [10, 3, 2, 4, 96, 192, 1, 0.25],
                    [19, 3, 1, 6, 192, 224, 1, 0.25],
                    [25, 3, 2, 6, 224, 384, 1, 0.25],
                    [7, 3, 1, 6, 384, 640, 1, 0.25]]

    return EfficientNetV2(model_cnf=model_config,
                          num_classes=num_classes,
                          dropout_rate=0.4)
