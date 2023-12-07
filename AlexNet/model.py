# _*_ coding: utf-8 _*_

"""
  @Time    : 2023.11.26 08:30
  @File    : model.py
  @Project : CNN_Demo
  @Author  : Saitama
  @IDE     : PyCharm
  @Desc    : CNN Demo 【AlexNet】
             用 Pytorch 搭建 AlexNet
             在全连接层，使用dropout，使一部分神经元失活，以解决过拟合问题
"""

import torch
import torch.nn as nn


class AlexNet(nn.Module):
    def __init__(self, num_classes=1000, init_weights=False):
        super(AlexNet, self).__init__()
        # 将整个网络进行打包，精简代码
        self.features = nn.Sequential(
            # input - [3, 224, 224]
            nn.Conv2d(3, 48, 11, 4, 2),  # output - [48, 55, 55]
            nn.ReLU(inplace=True),  # inplace：通过此种设置，可以在内存中载入更大的模型
            nn.MaxPool2d(3, 2),  # output - [48, 27, 27]
            nn.Conv2d(48, 128, 5, 1, 2),  # output - [128, 27, 27]
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, 2),  # output - [128, 13, 13]
            nn.Conv2d(128, 192, 3, 1, 1),  # output - [192, 13, 13]
            nn.ReLU(inplace=True),
            nn.Conv2d(192, 192, 3, 1, 1),  # output - [192, 13, 13]
            nn.ReLU(inplace=True),
            nn.Conv2d(192, 128, 3, 1, 1),  # output - [128, 13, 13]
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, 2),  # output - [128, 6, 6]
        )

        self.classifier = nn.Sequential(
            nn.Dropout(0.5),  # 随机使50%的神经元失活
            nn.Linear(128 * 6 * 6, 2048),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(2048, 2048),
            nn.ReLU(inplace=True),
            nn.Linear(2048, num_classes),
        )

        if init_weights:
            self.__initialize_weights()

    def __initialize_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal(module.weight, mode='fan_out', nonlinearity='relu')
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.Linear):
                nn.init.normal(module.weight, 0, 0.01)
                nn.init.constant_(module.bias, 0)

    # 正向传播
    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, start_dim=1)  # 因为Pytorch中的Tensor维度为 [batch, channel, height, width]，所以从channel开始
        x = self.classifier(x)
        return x
