# _*_ coding: utf-8 _*_

"""
  @Time    : 2023.12.22 09:42
  @File    : model.py
  @Project : cnn_demo
  @Author  : Saitama
  @IDE     : PyCharm
  @Desc    : 手搓ShuffleNet
            模型定义
"""
from typing import List, Callable

import torch
from torch import Tensor, nn


def channel_shuffle(x: Tensor, groups: int) -> Tensor:
    """
    将特征矩阵中的channel进行混合
    :param x:
    :param groups:
    :return:
    """
    batch_size, num_channels, height, width = x.size()
    channels_per_group = num_channels // groups

    # reshape
    x = x.view(batch_size, groups, channels_per_group, height, width)

    x = torch.transpose(x, 1, 2).contiguous()

    # flatten
    x = x.view(batch_size, -1, height, width)

    return x


class InvertedResidual(nn.Module):
    def __init__(self, input_c: int, output_c: int, stride: int):
        super(InvertedResidual, self).__init__()

        if stride not in [1, 2]:
            raise ValueError("illegal stride value")
        self.stride = stride

        # 确保 channel可以均分
        assert output_c % 2 == 0

        # 每个分支的channel
        branch_features = output_c // 2

        # 当stride为1时，input_channel应该是branch_features的两倍
        assert (self.stride != 1) or (input_c == branch_features << 1)

        if self.stride == 2:
            self.branch_left = nn.Sequential(
                self.depthwise_conv(input_c, input_c, kernel_s=3, stride=2, padding=1),
                nn.BatchNorm2d(input_c),
                nn.Conv2d(input_c, branch_features, kernel_size=1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(branch_features),
                nn.ReLU(inplace=True)
            )
        else:
            # stride 为1时，左边的branch不做任何处理
            self.branch_left = nn.Sequential()

        self.branch_right = nn.Sequential(
            # stride 为1时，input_channel 需要做 Channel Split 操作，所以应该除以2（即为branch_features）
            # stride 为2时，不需要做上述操作，所以不需要除以2
            nn.Conv2d(input_c if stride == 2 else branch_features, branch_features, kernel_size=1, stride=1, padding=0,
                      bias=False),
            nn.BatchNorm2d(branch_features),
            nn.ReLU(inplace=True),
            # DW 卷积的输入与输出的channel保持一致
            self.depthwise_conv(branch_features, branch_features, kernel_s=3, stride=stride, padding=1),
            nn.BatchNorm2d(branch_features),
            nn.Conv2d(branch_features, branch_features, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(branch_features),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        """
        正向传播
        :param x:
        :return:
        """
        if self.stride == 1:
            # stride 为1时，输入需要先进行分块【Channel Split】
            # dims=1 [batch, channel, height, width]
            x1, x2 = x.chunk(2, dims=1)
            out = torch.cat((x1, self.branch_right(x2)), dim=1)
        else:
            # stride 为2时，输入不需要先进行分块
            out = torch.cat((self.branch_left(x), self.branch_right(x)), dim=1)

        out = channel_shuffle(out, groups=2)

        return out

    @staticmethod
    def depthwise_conv(input_c: int,
                       output_c: int,
                       kernel_s: int,
                       stride: int = 1,
                       padding: int = 0,
                       bias: bool = False) -> nn.Conv2d:
        """
        DW 卷积
        输入特征矩阵的channel = 输出特征矩阵的channel
        :param input_c:
        :param output_c:
        :param kernel_s:
        :param stride:
        :param padding:
        :param bias:
        :return:
        """
        return nn.Conv2d(in_channels=input_c, out_channels=output_c, kernel_size=kernel_s, stride=stride,
                         padding=padding, bias=bias)


class ShuffleNetV2(nn.Module):
    """
    根据网络结构图，使用上述Block搭建 ShuffleNet-V2 网络
    """

    def __init__(self,
                 stage_repeats: List[int],
                 stage_out_channels: List[int],
                 num_classes: int = 1000,
                 inverted_residual: Callable[..., nn.Module] = InvertedResidual):
        """
        ShuffleNet-V2 初始化函数
        :param stage_repeats: 每个Stage中Block的重复次数，例如【1X版本，stage_repeats=[4, 8, 4]】
        :param stage_out_channels: 每个Stage的输出维度，例如【1X版本，stage_out_channels=[24, 116, 232, 464, 1024]】
        :param num_classes: 预测类别
        :param inverted_residual: 预先定义的Block
        """
        super(ShuffleNetV2, self).__init__()

        if len(stage_repeats) != 3:
            raise ValueError("expected stage_repeats as list of 3 int value")
        if len(stage_out_channels) != 5:
            raise ValueError("expected stage_out_channels as list fo 5 int value")

        # 输入图像为 RGB 图像
        input_channels = 3

        # Conv1的输出维度
        output_channels = stage_out_channels[0]

        # 第一个卷积层，对应结构图中的 Conv1
        self.Conv1 = nn.Sequential(
            nn.Conv2d(in_channels=input_channels, out_channels=output_channels, kernel_size=3, stride=2, padding=1,
                      bias=False),
            nn.BatchNorm2d(output_channels),
            nn.ReLU(inplace=True)
        )

        # 将第一个卷积的输出维度重新赋值给 input_channels，用于后续操作
        input_channels = output_channels

        # 最大池化层
        self.MP = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # 搭建 Stage 2 ~ 4
        stage_names = ["stage{}".format(i) for i in [2, 3, 4]]
        for name, repeats, output_channels in zip(stage_names, stage_repeats, stage_out_channels[1:4]):
            # 每个Stage的头部：stride为2的Block
            seq = [inverted_residual(input_channels, output_channels, 2)]
            for i in range(repeats - 1):
                # 每个Stage的尾部：stride为1的Block
                seq.append(inverted_residual(output_channels, output_channels, 1))
            setattr(self, name, nn.Sequential(*seq))
            # 每个Stage结束，需要将当前Stage的输出维度重新赋值给 input_channels，用于下个Stage使用
            input_channels = output_channels

        # Conv5的输入维度：input_channels
        # Conv5的输出维度：output_channels = stage_out_channels[-1]
        output_channels = stage_out_channels[-1]
        self.Conv5 = nn.Sequential(
            nn.Conv2d(in_channels=input_channels, out_channels=output_channels, kernel_size=1, stride=1,
                      padding=0, bias=False),
            nn.BatchNorm2d(output_channels),
            nn.ReLU(inplace=True)
        )

        # 全连接层
        self.FC = nn.Linear(in_features=output_channels, out_features=num_classes)

    def forward(self, x: Tensor) -> Tensor:
        """
        正向传播
        :param x:
        :return:
        """
        out = self.Conv1(x)
        out = self.MP(out)
        out = self.stage2(out)
        out = self.stage3(out)
        out = self.stage4(out)
        out = self.Conv5(out)
        out = out.mean([2, 3])  # 全局池化
        out = self.FC(out)
        return out


def generate_shufflenet_v2_x0_5(num_classes=1000):
    """
    快速生成 ShuffleNet-V2 的 0.5X 版本
    :param num_classes: 预测类别
    :return:
    """
    stage_repeats = [4, 8, 4]
    stage_out_channels = [24, 48, 96, 192, 1024]

    model = ShuffleNetV2(stage_repeats=stage_repeats,
                         stage_out_channels=stage_out_channels,
                         num_classes=num_classes)

    return model


def generate_shufflenet_v2_x1_0(num_classes=1000):
    """
    快速生成 ShuffleNet-V2 的 1X 版本
    :param num_classes: 预测类别
    :return:
    """
    stage_repeats = [4, 8, 4]
    stage_out_channels = [24, 116, 232, 464, 1024]

    model = ShuffleNetV2(stage_repeats=stage_repeats,
                         stage_out_channels=stage_out_channels,
                         num_classes=num_classes)

    return model


def generate_shufflenet_v2_x1_5(num_classes=1000):
    """
    快速生成 ShuffleNet-V2 的 1.5X 版本
    :param num_classes: 预测类别
    :return:
    """
    stage_repeats = [4, 8, 4]
    stage_out_channels = [24, 176, 352, 704, 1024]

    model = ShuffleNetV2(stage_repeats=stage_repeats,
                         stage_out_channels=stage_out_channels,
                         num_classes=num_classes)

    return model


def generate_shufflenet_v2_x2_0(num_classes=1000):
    """
    快速生成 ShuffleNet-V2 的 2.0X 版本
    :param num_classes: 预测类别
    :return:
    """
    stage_repeats = [4, 8, 4]
    stage_out_channels = [24, 244, 488, 976, 2048]

    model = ShuffleNetV2(stage_repeats=stage_repeats,
                         stage_out_channels=stage_out_channels,
                         num_classes=num_classes)

    return model
