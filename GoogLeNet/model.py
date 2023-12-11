# _*_ coding: utf-8 _*_

"""
  @Time    : 2023.12.08 11:28
  @File    : model.py
  @Project : cnn_demo
  @Author  : Saitama
  @IDE     : PyCharm
  @Desc    : 手搓 GoogLeNet
            用 PyTorch 搭建 GoogLeNet
"""
import torch
import torch.nn.functional as F

from torch import nn


class GoogLeNet(nn.Module):
    def __init__(self, num_classes=10, aux_logits=True, init_weights=False):
        super(GoogLeNet, self).__init__()
        self.aux_logits = aux_logits

        self.conv1 = BasicConv2d(3, 64, kernel_size=7, stride=2, padding=3)
        self.maxpool1 = nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True)

        self.conv2 = BasicConv2d(64, 64, kernel_size=1, stride=1)
        self.conv3 = BasicConv2d(64, 192, kernel_size=3, stride=1, padding=1)
        self.maxpool2 = nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True)

        self.inception_3a = Inception(192, 64, 96, 128, 16, 32, 32)
        self.inception_3b = Inception(256, 128, 128, 192, 32, 96, 64)
        self.maxpool3 = nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True)

        self.inception_4a = Inception(480, 192, 96, 208, 16, 48, 64)
        self.inception_4b = Inception(512, 160, 112, 224, 24, 64, 64)
        self.inception_4c = Inception(512, 128, 128, 256, 24, 64, 64)
        self.inception_4d = Inception(512, 112, 144, 288, 32, 64, 64)
        self.inception_4e = Inception(528, 256, 160, 320, 32, 128, 128)
        self.maxpool4 = nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True)

        self.inception_5a = Inception(832, 256, 160, 320, 32, 128, 128)
        self.inception_5b = Inception(832, 384, 192, 384, 48, 128, 128)

        # 如果采用辅助分类器
        if self.aux_logits:
            self.aux_logits_1 = InceptionAux(512, num_classes)
            self.aux_logits_2 = InceptionAux(528, num_classes)

        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(p=0.4)
        self.fc = nn.Linear(1024, num_classes)

        if init_weights:
            self._initialize_weights()

    def _initialize_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal(module.weight, mode='fan_out', nonlinearity='relu')
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.Linear):
                nn.init.normal(module.weight, 0, 0.01)
                nn.init.constant_(module.bias, 0)

    def forward(self, x):
        """
        正向传播
        :param x:
        :return:
        """
        # input_size=[3, 224, 224]
        x = self.conv1(x)  # output_size=[]
        x = self.maxpool1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.maxpool2(x)

        x = self.inception_3a(x)
        x = self.inception_3b(x)

        x = self.maxpool3(x)

        x = self.inception_4a(x)
        if self.training and self.aux_logits:
            aux1 = self.aux_logits_1(x)
        x = self.inception_4b(x)
        x = self.inception_4c(x)
        x = self.inception_4d(x)
        if self.training and self.aux_logits:
            aux2 = self.aux_logits_2(x)
        x = self.inception_4e(x)

        x = self.maxpool4(x)

        x = self.inception_5a(x)
        x = self.inception_5b(x)

        x = self.avg_pool(x)

        x = torch.flatten(x, 1)

        x = self.dropout(x)
        x = self.fc(x)

        if self.training and self.aux_logits:
            return x, aux2, aux1
        return x


class Inception(nn.Module):
    """
    Inception module
    """

    def __init__(self, in_channels, ch1x1, ch3x3red, ch3x3, ch5x5red, ch5x5, pool_proj):
        super(Inception, self).__init__()

        # branch1 - ch1x1[BasicConv2d]
        self.branch1 = BasicConv2d(in_channels, ch1x1, kernel_size=1)

        # branch2 - ch3x3red[BasicConv2d] + ch3x3[BasicConv2d]
        self.branch2 = nn.Sequential(
            BasicConv2d(in_channels, ch3x3red, kernel_size=1),
            BasicConv2d(ch3x3red, ch3x3, kernel_size=3, padding=1)
        )

        # branch3 - ch5x5red[BasicConv2d] + ch5x5[BasicConv2d]
        self.branch3 = nn.Sequential(
            BasicConv2d(in_channels, ch5x5red, kernel_size=1),
            BasicConv2d(ch5x5red, ch5x5, kernel_size=5, padding=2)
        )

        # branch4 - 3x3Maxpool + pool_proj[BasicConv2d]
        self.branch4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            BasicConv2d(in_channels, pool_proj, kernel_size=1)
        )

    def forward(self, x):
        """
        正向传播【对四个分支的结果在深度（channel）方向进行合并】
        :param x:
        :return:
        """
        output1 = self.branch1(x)
        output2 = self.branch2(x)
        output3 = self.branch3(x)
        output4 = self.branch4(x)

        outputs = [output1, output2, output3, output4]

        # 因为PyTorch中Tensor的通道为：[batch, channel, height, width]
        return torch.cat(outputs, 1)


class InceptionAux(nn.Module):
    """
    InceptionAux module【辅助分类器】
    """

    def __init__(self, in_channels, num_classes):
        super(InceptionAux, self).__init__()
        self.average_pool = nn.AvgPool2d(kernel_size=5, stride=3)
        self.conv = BasicConv2d(in_channels, 128, kernel_size=1)  # output[batch, 128, 4, 4]

        self.fc1 = nn.Linear(2048, 1024)
        self.fc2 = nn.Linear(1024, num_classes)

    def forward(self, x):
        """
        正向传播
        :param x:
        :return:
        """
        # aux1_input_size: [512, 14, 14]
        # aux2_input_size: [528, 14, 14]
        x = self.average_pool(x)
        # After Average_Pool
        # aux1_output_size: [512, 4, 4]
        # aux2_output_size: [528, 4, 4]
        x = self.conv(x)
        # After Conv
        # aux1_output_size & aux2_output_size: [128, 4, 4]
        x = torch.flatten(x, 1)
        # After Flatten
        # aux1_output_size & aux2_output_size: [2048]

        # Dropout + FC1 + RELU
        x = F.dropout(x, 0.5, training=self.training)
        x = self.fc1(x)  # output_size: [1024]
        x = F.relu(x, inplace=True)

        # Dropout + FC2
        x = F.dropout(x, 0.5, training=self.training)
        x = self.fc2(x)  # output_size: [num_classes]

        return x


class BasicConv2d(nn.Module):
    """
    BasicConv2d [Conv2d + RELU]
    """

    def __init__(self, in_channels, out_channels, **kwargs):
        super(BasicConv2d, self).__init__()
        self.conv2d = nn.Conv2d(in_channels, out_channels, **kwargs)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        """
        正向传播【卷积 + RELU】
        :param x:
        :return:
        """
        x = self.conv2d(x)
        x = self.relu(x)
        return x
