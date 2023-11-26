# _*_ coding: utf-8 _*_

"""
  @Time    : 2023.11.26 11:51
  @File    : train.py
  @Project : CNN_Demo
  @Author  : Saitama
  @IDE     : PyCharm
  @Desc    : CNN Demo 【AlexNet】
             训练代码
"""
import json
import os.path
import time

import torch
import torch.nn as nn
import torch.utils.data
from torch import optim
from torchvision import transforms, datasets

from AlexNet.model import AlexNet

# 获取GPU&CPU
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# 定义数据预处理函数
data_transform = {
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

batch_size = 32

# 图片根目录
data_root = os.path.abspath(os.path.join(os.getcwd(), "./data/flower_photos"))

# 训练集
train_dataset = datasets.ImageFolder(root=data_root + "/train", transform=data_transform["train"])
train_num = len(train_dataset)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

# 验证集
validated_dataset = datasets.ImageFolder(root=data_root + "/val", transform=data_transform["val"])
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

# 实例化AlexNet网络
net = AlexNet(num_classes=5, init_weights=True)

# 将网络部署到GPU上
net.to(device)

# 定义损失函数
loss_function = nn.CrossEntropyLoss()

# 定义优化器(优化对象是网络中所有的参数)
optimizer = optim.Adam(net.parameters(), lr=0.0002)

# 权重文件保存路径
save_path = "./weights/AlexNet.pth"

# 最佳准确率
best_acc = 0.0

for epoch in range(10):
    # train
    net.train()  # net.train()则会启用dropout【不希望在eval中使用dropout，因为会影响模型的效果】
    # 单次epoch中累积损失
    running_loss = 0.0
    t1 = time.perf_counter()  # 用于计时
    for step, data in enumerate(train_loader, start=0):
        images, labels = data
        # 将历史损失梯度清零
        optimizer.zero_grad()
        # 进行正向传播(指派到GPU)
        outputs = net(images.to(device))
        # 将预测结果与真实标签对比，计算损失
        loss = loss_function(outputs, labels.to(device))
        # 将损失进行反向传播
        loss.backward()
        # 进行参数更新
        optimizer.step()

        # 更新当前epoch的累积损失
        running_loss += loss.item()

        # 实时打印结果
        rate = (step + 1) / len(train_loader)
        a = "*" * int(rate * 50)
        b = "." * int((1 - rate) * 50)
        print("\rtrain loss: {:^3.0f}%[{}->{}]{:.3f}".format(int(rate * 100), a, b, loss), end="")
    print()
    print(time.perf_counter() - t1)

    # validate
    net.eval()
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
        accuracy_cur_epoch = acc / val_num
        if accuracy_cur_epoch > best_acc:
            best_acc = accuracy_cur_epoch
            torch.save(net.state_dict(), save_path)
        print("[epoch %d] train loss: %.3f test_accuracy %.3f best_accuracy %.3f" %
              (epoch + 1, running_loss / step, accuracy_cur_epoch, best_acc))

print("Training done !!!")