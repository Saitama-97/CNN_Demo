# _*_ coding: utf-8 _*_

"""
  @Time    : 2023.11.26 22:08
  @File    : predict.py
  @Project : CNN_Demo
  @Author  : Saitama
  @IDE     : PyCharm
  @Desc    : CNN Demo 【AlexNet】
             预测脚本
"""
import json

import matplotlib.pyplot as plt
import torch
from PIL import Image
import torchvision.transforms as transforms

from AlexNet.model import AlexNet

# 图片预处理函数(resize + 通道变换 + 归一化)
transform = transforms.Compose(
    [transforms.Resize((224, 224)),
     transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.9, 0.5, 0.5))]
)

# 加载图片
img = Image.open("./data/test1.jpg")
plt.imshow(img)
img = transform(img)
img = torch.unsqueeze(img, dim=0)

# 加载类别列表
try:
    json_file = open("./data/flower_photos/class_indexes.json", "r")
    classes = json.load(json_file)
except Exception as e:
    print(e)
    exit(-1)

# 初始化网络
net = AlexNet()
# 加载模型
net.load_state_dict(torch.load("./weights/AlexNet.pth"))

# 关闭dropout
net.eval()

with torch.no_grad():
    output = net(img)
    predict = torch.squeeze(output)
    predict_cls = torch.argmax(predict).numpy()

print(classes[str(predict_cls)], predict[predict_cls].item())
plt.show()