# _*_ coding: utf-8 _*_

"""
  @Time    : 2023.12.11 10:46
  @File    : train.py
  @Project : cnn_demo
  @Author  : Saitama
  @IDE     : PyCharm
  @Desc    : GoogLeNet 训练脚本
"""
import json
import os
import torch
from torch import optim

from torchvision import transforms, datasets

import torch.utils.data
import torch.nn as nn

from GoogLeNet.model import GoogLeNet

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

data_transforms = {
    "train": transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ]),
    "val": transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
}

# 图片根目录
data_root = os.path.abspath(os.path.join(os.getcwd(), "../AlexNet/data/flower_photos"))

batch_size = 32

# 训练集
train_dataset = datasets.ImageFolder(root=data_root + "/train", transform=data_transforms["train"])
train_num = len(train_dataset)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

# 验证集
validated_dataset = datasets.ImageFolder(root=data_root + "/val", transform=data_transforms["val"])
val_num = len(validated_dataset)
val_loader = torch.utils.data.DataLoader(validated_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

# 构建 class-index 对应关系
flower_list = train_dataset.class_to_idx
cls_dict = dict((val, key) for key, val in flower_list.items())

# 将对应关系保存为json文件
json_str = json.dumps(cls_dict, indent=4)
json_path = data_root + "/class_indexes.json"
with open(json_path, "w") as json_file:
    json_file.write(json_str)

# 实例化GoogLeNet网络
net = GoogLeNet(num_classes=5, aux_logits=True, init_weights=True)

# 部署到GPU
net.to(device)

# 定义损失函数
loss_func = nn.CrossEntropyLoss()

# 定义优化器
optimizer = optim.Adam(net.parameters(), lr=0.0003)

# 权重文件保存路径
save_path = "./weights/GoogLeNet.pth"

# 最佳准确率
best_acc = 0.0

for epoch in range(30):
    net.train()
    # 单次epoch中累积损失
    running_loss = 0.0
    for step, data in enumerate(train_loader, start=0):
        images, labels = data
        # 首先将历史损失梯度清零
        optimizer.zero_grad()
        # 进行正向传播(有三个输出【包含了两个辅助输出】)
        logits, aux_logits2, aux_logits1 = net(images.to(device))
        # 将三个预测结果与真实标签对比，计算损失
        loss0 = loss_func(logits, labels.to(device))
        loss1 = loss_func(aux_logits1, labels.to(device))
        loss2 = loss_func(aux_logits2, labels.to(device))

        # 综合计算损失
        loss = loss0 + loss1 * 0.3 + loss2 * 0.3

        # 将损失进行反向传播
        loss.backward()

        # 优化器进行参数更新
        optimizer.step()

        # 更新当前epoch的累计损失
        running_loss += loss

        # 实时打印结果
        rate = (step + 1) / len(train_loader)
        a = "*" * int(rate * 50)
        b = "." * int((1 - rate) * 50)
        print("\rtrain loss: {:^3.0f}%[{}->{}]{:.3f}".format(int(rate * 100), a, b, loss), end="")
    print()

    # validate
    net.eval()
    acc = 0.0
    with torch.no_grad():
        for val_data in val_loader:
            val_images, val_labels = val_data
            # 对验证集进行预测【此时不需要辅助分类器】
            val_outputs = net(val_images.to(device))
            predict_y = torch.max(val_outputs, dim=1)[1]
            # 根据预测结果和实际标签，计算准确率
            acc += (predict_y == val_labels.to(device)).sum().item()
        accuracy_cur_epoch = acc / val_num
        if accuracy_cur_epoch > best_acc:
            best_acc = accuracy_cur_epoch
            torch.save(net.state_dict(), save_path)
        print("[epoch %d] train loss: %.3f test_accuracy %.3f best_accuracy %.3f" %
              (epoch + 1, running_loss / step, accuracy_cur_epoch, best_acc))

print("Training Done !!!")