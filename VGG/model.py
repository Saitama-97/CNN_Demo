# _*_ coding: utf-8 _*_

"""
  @Time    : 2023.12.07 14:53
  @File    : model.py
  @Project : cnn_demo
  @Author  : Saitama
  @IDE     : PyCharm
  @Desc    : 手搓 VGG[A B D E]
            【特征提取网络 + 分类网络】
"""
import torch
import torch.nn as nn


class VGG(nn.Module):
    def __init__(self, features, class_num=1000, init_weights=False):
        """
        初始化 VGG
        :param features: 特征提取网络
        :param class_num: 类别数
        :param init_weights: 是否初始化权重
        """
        super(VGG, self).__init__()
        # VGG 分类网络
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(True),
            nn.Dropout(p=0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Linear(4096, class_num)
        )
        if init_weights:
            self._initialize_weights()

    def _initialize_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.xavier_uniform(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.Linear):
                nn.init.xavier_uniform(module.weight)
                nn.init.constant_(module.bias, 0)

    def forward(self, x):
        # input_size = [3,224,224]
        x = self.features(x)  # 特征提取【conv+maxpool】后，output_size = [512,7,7]
        x = torch.flatten(x, start_dim=1)  # 展平
        x = self.forward(x)  # 送入分类网络
        return x


# VGG 特征提取网络结构配置[A B D E]
cfgs = {
    # 数字表示卷积核个数，'M'表示最大池化层
    "vgg11": [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    "vgg13": [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    "vgg16": [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    "vgg19": [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512,
              'M'],
}


def make_feature(cfg: list):
    """
    根据传入的 VGG 特征提取网络配置参数，生成对应的 VGG 特征提取网络
    :param cfg:
    :return:
    """
    layers = []
    in_channels = 3
    for item in cfg:
        if item == 'M':
            mp = nn.MaxPool2d(kernel_size=2, stride=2)
            layers.append(mp)
        else:
            conv = nn.Conv2d(in_channels=3, out_channels=item, kernel_size=3, stride=1, padding=1)
            layers.append(conv)
            layers.append(nn.ReLU(True))  # inplace=True，可以载入更大的模型
            in_channels = item
    return nn.Sequential(*layers)  # *表示以非关键字参数传入Sequential


def vgg(model_name="vgg16", **kwargs):
    try:
        cfg = cfgs[model_name]
    except:
        print("model {} doesn't exist".format(model_name))
    model = VGG(features=make_feature(cfg), class_num=1000, init_weights=True)
    return model


if __name__ == '__main__':
    vgg_model = vgg(model_name="vgg16")
    print()
