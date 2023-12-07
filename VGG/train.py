# _*_ coding: utf-8 _*_

"""
  @Time    : 2023.12.07 15:52
  @File    : train.py
  @Project : cnn_demo
  @Author  : Saitama
  @IDE     : PyCharm
  @Desc    : 手搓 VGG[A B D E]
            训练模型
"""
import json
import os

import torch
import torch.nn as nn
import torch.utils.data
from torch import optim
from torchvision import transforms, datasets

from VGG.model import vgg

# 获取GPU&CPU
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# 定义数据预处理函数
data_transforms = {
    "train": transforms.Compose(
        [transforms.RandomResizedCrop(224),
         transforms.RandomHorizontalFlip(),
         transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    ),
    "val": transforms.Compose(
        [transforms.Resize((224, 224)),
         transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    )
}

# 图片根目录
data_root = os.path.abspath(os.path.join(os.getcwd(), "../AlexNet/data/flower_photos"))

# 训练集路径
train_path = data_root + "/train"
# 测试集路径
val_path = data_root + "/val"

# 包装数据集
train_dataset = datasets.ImageFolder(root=train_path, transform=data_transforms["train"])
train_nums = len(train_dataset)
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=32)

val_dataset = datasets.ImageFolder(root=val_path, transform=data_transforms["val"])
val_nums = len(val_dataset)
val_loader = torch.utils.data.DataLoader(dataset=val_dataset, batch_size=32)

# 构建 class-index 对应关系
flower_list = train_dataset.class_to_idx
cls_dict = dict((val, key) for key, val in flower_list.items())

# 将对应关系保存为json文件
json_str = json.dumps(cls_dict, indent=4)
json_path = data_root + "/class_indexes.json"
with open(json_path, "w") as json_file:
    json_file.write(json_str)

# 实例化 VGG-16 网络，准备训练
name = "vgg16"
net = vgg(model_name=name, class_nums=5, init_weights=True)

# 部署到设备上
net.to(device)

# 定义损失函数
loss_func = nn.CrossEntropyLoss()

# 定义优化器
optimizer = optim.Adam(net.parameters(), lr=0.0001)

# 权重文件保存路径
save_path = "./weights/{}.pth".format(name)

# 最佳准确率
best_acc = 0.0

for epoch in range(30):
    net.train()  # 即可启用dropout，在eval中不要使用dropout
    # 单次epoch的累计损失
    running_loss = 0.0
    for step, data in enumerate(train_loader, start=0):
        images, labels = data
        # 将历史损失梯度清零
        optimizer.zero_grad()
        # 进行正向传播
        outputs = net(images.to(device))
        # 计算损失
        loss = loss_func(outputs, labels.to(device))
        # 将损失进行反向传播
        loss.backward()
        # 对优化器进行参数更新
        optimizer.step()

        # 更新当前epoch的累积损失
        running_loss += loss.item()

        # 实时打印结果
        rate = (step + 1) / len(train_loader)
        a = "*" * int(rate * 50)
        b = "." * int((1 - rate) * 50)
        print("\rtrain loss: {:^3.0f}%[{}->{}]{:.3f}".format(int(rate * 100), a, b, loss), end="")

    # validate
    net.eval()  # 不启用dropout
    acc = 0.0
    with torch.no_grad():
        for val_data in val_loader:
            val_images, val_labels = val_data
            # 对验证集进行预测
            val_outputs = net(val_images.to(device))
            # 每个样本取最可能的类别
            predict_y = torch.max(val_outputs, dim=1)[1]
            # 根据预测结果和实际标签，计算准确率
            acc += (predict_y == val_labels.to(device)).sum().item()
        accuracy_cur_epoch = acc / val_nums
        if accuracy_cur_epoch > best_acc:
            best_acc = accuracy_cur_epoch
            torch.save(net.state_dict(), save_path)
        print("[epoch %d] train loss: %.3f test_accuracy %.3f best_accuracy %.3f" %
              (epoch + 1, running_loss / step, accuracy_cur_epoch, best_acc))

print("Training done !!!")
