# _*_ coding: utf-8 _*_

"""
  @Time    : 2023.12.20 17:39
  @File    : model.py
  @Project : cnn_demo
  @Author  : Saitama
  @IDE     : PyCharm
  @Desc    : GoogleNet 手搓
            模型定义
"""

import torch
import torch.nn as nn


class ConvBNRelu(nn.Sequential):
    def __init__(self, in_channel, out_channel, kernel_size=3, stride=1, groups=1):
        padding = (kernel_size - 1) // 2
        super(ConvBNRelu).__init__(
            nn.Conv2d(in_channel, out_channel, kernel_size, stride, padding, groups, bias=False),
            nn.BatchNorm2d(out_channel),
            nn.ReLU6(inplace=True)
        )


"""
TBD
"""

