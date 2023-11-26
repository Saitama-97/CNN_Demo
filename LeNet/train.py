# _*_ coding: utf-8 _*_

"""
  @Time    : 2023.11.24 10:48
  @File    : train.py
  @Project : CNN_demo
  @Author  : Saitama
  @IDE     : PyCharm
  @Desc    : CNN-Pytorch 官方Demo【LeNet+cifar10】
             训练
"""
import matplotlib.pyplot as plt
import numpy as np
import torch.utils.data
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch import optim

from LeNet.model import LeNet

# 对图像进行预处理的函数
# ToTensor：(HWC)->(CHW)
# Normalize：标准化
transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
)

# 下载训练集(50000张图片)
train_set = torchvision.datasets.CIFAR10(root="./data", train=True, transform=transform, download=False)

# 加载训练数据(Windows下num_workers应为0)
train_loader = torch.utils.data.DataLoader(train_set, batch_size=36, shuffle=True, num_workers=0)

# 加载测试集(10000张图片)
test_set = torchvision.datasets.CIFAR10(root="./data", train=False, transform=transform, download=False)

test_loader = torch.utils.data.DataLoader(test_set, batch_size=10000, shuffle=True, num_workers=0)

# print()
# 将测试集转化成一个迭代器
test_data_iter = iter(test_loader)
# 测试图片，测试标签
test_image, test_label = test_data_iter.next()

classes = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

# 初始化 LeNet 网络
net = LeNet()
# 损失函数
loss_function = nn.CrossEntropyLoss()
# 优化器
optimizer = optim.Adam(net.parameters(), lr=0.001)

for epoch in range(20):
    # 每个epoch的累积损失
    running_loss = 0.0
    for step, data in enumerate(train_loader, start=0):
        # 原图，标签
        inputs, labels = data

        # 将历史损失梯度清零【如果不清零，则会对历史梯度进行累加，当然也可以通过这个特性，实现一个很大batch值的训练】
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(inputs)  # 当前的预测结果
        loss = loss_function(outputs, labels)  # 将当前的预测结果与真实标签进行对比，计算损失
        loss.backward()  # 反向传播
        optimizer.step()  # 进行参数更新

        # 更新当前epoch的累积损失
        running_loss += loss.item()

        # 每500步打印一次结果
        if step % 500 == 499:
            # 当前测试阶段，不计算梯度，否则会占用大量计算资源（时间+空间）
            with torch.no_grad():
                # 对测试集进行预测
                outputs = net(test_image)
                # 每个样本取最可能的类别
                predict_y = torch.max(outputs, dim=1)[1]
                # 计算准确率
                accuracy = (predict_y == test_label).sum().item() / test_label.size()[0]
                # 打印结果
                print("[%d, %5d] train_loss: %.3f test_accuracy: %.3f" % (
                    epoch + 1, step + 1, running_loss / 500, accuracy))
                # 一轮epoch结束，将累积损失清零
                running_loss = 0.0

print("Training Complete!!!")

save_path = "./weights/LeNet.pth"
torch.save(net.state_dict(), save_path)
