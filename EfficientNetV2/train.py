# _*_ coding: utf-8 _*_

"""
  @Time    : 2023.12.26 16:56
  @File    : train.py
  @Project : CNN_Demo
  @Author  : Saitama
  @IDE     : PyCharm
  @Desc    : 手搓 EfficientNet-V2
            训练模型
"""

import argparse
import math
import os.path

import torch
from torch import optim
from torch.optim import lr_scheduler
from torchvision import transforms
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader

from EfficientNetV2.model import generate_efficientnetv2_s
from my_dataset import MyDataSet
from utils import train_one_epoch, evaluate, read_split_data


def main(args):
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    print(args)

    print("Start Tensorboard with tensorboard --logdir=runs, view at http://localhost:6006/")

    tb_writer = SummaryWriter()

    if not os.path.exists("./weights"):
        os.mkdir("./weights")

    train_images_path, train_images_label, val_images_path, val_images_label = read_split_data(args.data_path)

    # train_size, eval_size
    img_size = {"s": [300, 384],
                "m": [384, 480],
                "l": [384, 480],
                }

    num_model = "s"

    # 数据预处理函数
    data_transform = {
        "train": transforms.Compose([transforms.RandomResizedCrop(img_size[num_model][0]),
                                     transforms.RandomHorizontalFlip(),
                                     transforms.ToTensor(),
                                     transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])]),
        "val": transforms.Compose([transforms.Resize(img_size[num_model][1]),
                                   transforms.CenterCrop(img_size[num_model][1]),
                                   transforms.ToTensor(),
                                   transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])}

    # 训练用数据集
    train_dataset = MyDataSet(images_path=train_images_path,
                              images_class=train_images_label,
                              transform=data_transform["train"])

    # 验证用数据集
    val_dataset = MyDataSet(images_path=val_images_path,
                            images_class=val_images_label,
                            transform=data_transform["val"])

    batch_size = args.batch_size

    # num_workers = min(os.cpu_count(), batch_size if batch_size > 1 else 0, 8)
    num_workers = 0

    print("Using {} dataloader workers every process".format(num_workers))

    # 训练用数据加载器
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                               batch_size=batch_size,
                                               shuffle=True,
                                               num_workers=num_workers,
                                               collate_fn=train_dataset.collate_fn)

    # 验证用数据加载器
    val_loader = torch.utils.data.DataLoader(dataset=val_dataset,
                                             batch_size=batch_size,
                                             shuffle=True,
                                             num_workers=num_workers,
                                             collate_fn=val_dataset.collate_fn)

    # 实例化网络模型
    model = generate_efficientnetv2_s(num_classes=args.num_classes).to(device)

    # 如果有预训练权重则载入
    if args.weights != "":
        if os.path.exists(args.weights):
            weights_dict = torch.load(args.weights, map_location=device)
            load_weights_dict = {k: v for k, v in weights_dict.items()
                                 if model.state_dict()[k].numel() == v.numel()}
            print(model.load_state_dict(load_weights_dict, strict=False))
        else:
            raise FileNotFoundError("not found weights file: {}".format(args.weights))

    # 是否冻结权重
    if args.freeze_layers:
        for name, para in model.named_parameters():
            # 除最后一个卷积层以及全连接层外，其他权重全部冻结
            if "head" not in name:
                para.requires_grad_(False)
            else:
                print("training {}".format(name))

    # 获取非冻结层的所有参数
    paras = [parameter for parameter in model.parameters() if parameter.requires_grad]

    # 定义优化器
    optimizer = optim.SGD(params=paras,
                          lr=args.lr,
                          momentum=0.9,
                          weight_decay=1E-4)

    lf = lambda x: ((1 + math.cos(x * math.pi / args.epochs)) / 2) * (1 - args.lrf) + args.lrf  # cosine
    scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)

    best_acc = 0

    for epoch in range(args.epochs):
        # train
        mean_loss = train_one_epoch(model=model,
                                    optimizer=optimizer,
                                    data_loader=train_loader,
                                    device=device,
                                    epoch=epoch)

        scheduler.step()

        # val
        acc = evaluate(model=model,
                       data_loader=val_loader,
                       device=device)

        print("[epoch {}] accuracy: {}".format(epoch, round(acc, 3)))

        tags = ["loss", "accuracy", "learning_rate"]

        tb_writer.add_scalar(tags[0], mean_loss, epoch)
        tb_writer.add_scalar(tags[1], acc, epoch)
        tb_writer.add_scalar(tags[2], optimizer.param_groups[0]["lr"], epoch)

        torch.save(model.state_dict(), "./weights/model_{}.pth".format(epoch))

        if acc > best_acc:
            best_acc = acc
            torch.save(model.state_dict(), "./weights/model_best.pth")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--num_classes', type=int, default=5)
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--lrf', type=float, default=0.01)

    # 数据集所在根目录
    parser.add_argument('--data-path', type=str,
                        default="../ShuffleNet/data/flower_photos")

    # EfficientNetV2_s 官方权重下载地址
    # 链接: https://pan.baidu.com/s/1uZX36rvrfEss-JGj4yfzbQ  密码: 5gu1
    parser.add_argument('--weights', type=str,
                        default='./weights/efficientnetv2-s.pth',
                        help='initial weights path')
    parser.add_argument('--freeze-layers', type=bool, default=False)
    parser.add_argument('--device', default='cuda:0', help='device id (i.e. 0 or 0,1 or cpu)')

    opts = parser.parse_args()

    main(opts)
