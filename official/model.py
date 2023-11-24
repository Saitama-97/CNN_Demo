# _*_ coding: utf-8 _*_

"""
  @Time    : 2023.11.24 10:50
  @File    : model.py
  @Project : cnn_demo
  @Author  : Saitama
  @IDE     : PyCharm
  @Desc    : CNN-Pytorch 官方Demo【LeNet+cifar10】
             定义 LeNet
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        # 第一个卷积层
        # 卷积之后，尺寸为 N=(W-F+2P)/S+1
        self.conv1 = nn.Conv2d(3, 16, 5)
        # 第一个池化层
        self.pool1 = nn.MaxPool2d(2, 2)

        # 第二个卷积层
        self.conv2 = nn.Conv2d(16, 32, 5)
        # 第二个池化层
        self.pool2 = nn.MaxPool2d(2, 2)

        # 第一个全连接层
        self.fc1 = nn.Linear(32 * 5 * 5, 120)
        # 第二个全连接层
        self.fc2 = nn.Linear(120, 84)
        # 第三个全连接层(此处的out_features取决于具体的数据集)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        # input (3, 32, 32) 【channel, height, width】
        x = F.relu(self.conv1(x))  # 第一次卷积之后尺寸为 (32-5+2*0)/1+1=28 (16, 28, 28)
        x = self.pool1(x)  # 第一次池化之后，尺寸为 (16, 14, 14)
        x = F.relu(self.conv2(x))  # 第二次卷积之后尺寸为 (14-5+2*0)/1+1=10 (32, 10, 10)
        x = self.pool2(x)  # 第二次池化之后，尺寸为 (32, 5, 5)
        x = x.view(-1, 32 * 5 * 5)  # 将数据展平
        x = F.relu(self.fc1(x))  # 第一次全连接之后，尺寸为 (120)
        x = F.relu(self.fc2(x))  # 第二次全连接之后，尺寸为 (84)
        x = F.relu(self.fc3(x))  # 第三次全连接之后，尺寸为 (10)
        return x


input1 = torch.rand([32, 3, 32, 32])
model = LeNet()
print(model)

output = model(input1)
