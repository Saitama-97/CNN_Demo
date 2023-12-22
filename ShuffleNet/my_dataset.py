# _*_ coding: utf-8 _*_

"""
  @Time    : 2023.12.22 15:37
  @File    : my_dataset.py
  @Project : cnn_demo
  @Author  : Saitama
  @IDE     : PyCharm
  @Desc    : 自定义数据集
"""
import torch

from PIL import Image
from torch.utils.data import Dataset


class MyDataSet(Dataset):
    def __init__(self,
                 images_path: list,
                 images_class: list,
                 transform=None):
        self.images_path = images_path
        self.images_class = images_class
        self.transform = transform

    def __len__(self):
        return len(self.images_path)

    def __getitem__(self, item):
        img = Image.open(self.images_path[item])
        if img.mode != "RGB":  # 灰度图
            raise ValueError("image: {} isn't RGB mode.".format(self.images_path[item]))
        label = self.images_class[item]

        if self.transform is not None:
            img = self.transform(img)

        return img, label

    @staticmethod
    def collate_fn(batch):
        # 官方实现的default_collate可以参考
        # https://github.com/pytorch/pytorch/blob/67b7e751e6b5931a9f45274653f4f653a4e6cdf6/torch/utils/data/_utils/collate.py
        images, labels = tuple(zip(*batch))

        images = torch.stack(images, dim=0)
        labels = torch.as_tensor(labels)
        return images, labels
