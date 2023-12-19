# _*_ coding: utf-8 _*_

"""
  @Time    : 2023/12/19 14:47 
  @File    : model.py
  @Project : CNN_Demo
  @Author  : Saitama
  @IDE     : PyCharm
  @Des     : 手搓ResNet
            模型定义
"""
import torch
import torch.nn as nn


class BasicBlock(nn.Module):
    """
    18、34层
    """
    expansion = 1

    # down_sample：是否有虚线【多层残差网络中使用 - 50、101、152】
    def __init__(self, in_channel, out_channel, stride=1, down_sample=None):
        super(BasicBlock, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=in_channel, out_channels=out_channel, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(num_features=out_channel)

        self.conv2 = nn.Conv2d(in_channels=out_channel, out_channels=out_channel, kernel_size=3, stride=1,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(num_features=out_channel)

        self.relu = nn.ReLU(inplace=True)

        self.down_sample = down_sample

    def forward(self, x):
        """
        正向传播
        :param x:
        :return:
        """
        identity = x

        # 如果下采样函数不为空，即为多层残差网络，需要首先对输入进行下采样
        if self.down_sample is not None:
            identity = self.down_sample(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out += identity
        out = self.relu(out)

        return out


class BottleNeck(nn.Module):
    """
    50、101、152层
    """
    expansion = 4

    def __init__(self, in_channel, out_channel, stride=1, down_sample=None):
        super(BottleNeck, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=in_channel, out_channels=out_channel, kernel_size=3, stride=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channel)

        self.conv2 = nn.Conv2d(in_channels=out_channel, out_channels=out_channel, kernel_size=3, stride=stride,
                               bias=False)
        self.bn2 = nn.BatchNorm2d(out_channel)

        self.conv3 = nn.Conv2d(in_channels=out_channel, out_channels=out_channel * self.expansion, kernel_size=1,
                               stride=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channel * self.expansion)

        self.relu = nn.ReLU(inplace=True)
        self.down_sample = down_sample

    def forward(self, x):
        identity = x
        if self.down_sample is not None:
            identity = self.down_sample(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        out += identity
        out = self.relu(out)

        return out


class ResNet(nn.Module):
    """
    残差神经网络定义
    """

    def __init__(self, block, blocks_nums, num_classes=1000, include_top=True):
        super(ResNet, self).__init__()
        self.include_top = include_top

        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)

        self.mp1 = nn.MaxPool2d(kernel_size=3, stride=2)

        self.layer1 = self._make_layer(block, 64, blocks_nums[0])
        self.layer2 = self._make_layer(block, 128, blocks_nums[1], stride=2)
        self.layer3 = self._make_layer(block, 256, blocks_nums[2], stride=2)
        self.layer4 = self._make_layer(block, 512, blocks_nums[3], stride=2)

        if self.include_top:
            self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
            self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal(m.weight, mode="fan_out")

    def _make_layer(self, block, channel, block_num, stride=1):
        """
        生成残差块
        :param block: 浅层（BasicBlock）还是深层(BottleNeck)
        :param channel: 第一层卷积核的个数
        :param block_num: 有几个残差结构
        :param stride:
        :return:
        """
        down_sample = None
        # 是否需要下采样（浅层【18、34】不需要，深层【50、101、152】需要）
        if stride != 1 or self.in_channel != channel * block.expansion:
            down_sample = nn.Sequential(
                nn.Conv2d(self.in_channel, channel * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(channel * block.expansion))

        layers = list()

        layers.append(block(in_channel=self.in_channel, out_channel=channel, stride=stride, down_sample=down_sample))
        self.in_channel = channel * block.expansion

        for _ in range(1, block_num):
            layers.append(block(in_channel=self.in_channel, out_channel=channel))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.mp1(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        if self.include_top:
            x = self.avgpool(x)
            x = torch.flatten(x, 1)
            x = self.fc(x)

        return x


def resnet_34(num_classes, include_top=True):
    return ResNet(BasicBlock, [3, 4, 6, 3], num_classes=num_classes, include_top=include_top)


def resnet_101(num_classes, include_top=True):
    return ResNet(BottleNeck, [3, 4, 23, 3], num_classes=num_classes, include_top=include_top)
