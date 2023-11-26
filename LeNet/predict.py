# _*_ coding: utf-8 _*_

"""
  @Time    : 2023.11.24 10:49
  @File    : predict.py
  @Project : CNN_demo
  @Author  : Saitama
  @IDE     : PyCharm
  @Desc    : CNN-Pytorch 官方Demo【LeNet+cifar10】
             预测
"""

from PIL import Image
import torch
import torchvision.transforms as transforms

from LeNet.model import LeNet

# 图片预处理函数(resize + 通道变换 + 归一化)
transform = transforms.Compose(
    [transforms.Resize((32, 32)),
     transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.9, 0.5, 0.5))]
)

# 类别
classes = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

# 初始化网络
net = LeNet()

# 加载模型
net.load_state_dict(torch.load("./weights/LeNet.pth"))

test_img = Image.open("./data/test1.png")  # [H, W, C]
test_img = transform(test_img)  # [C, H, W]
# pytorch 要求的Tensor: [batch, channel, height, width]
# 所以要加一个维度
test_img = torch.unsqueeze(test_img, dim=0)  # [N, C, H, W]

with torch.no_grad():
    output = net(test_img)
    predict = torch.max(output, dim=1)[1].data.numpy()

print(classes[int(predict)])
