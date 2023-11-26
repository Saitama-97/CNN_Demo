# _*_ coding: utf-8 _*_

"""
  @Time    : 2023.11.26 12:06
  @File    : split_data.py
  @Project : CNN_Demo
  @Author  : Saitama
  @IDE     : PyCharm
  @Desc    : 切分数据集（训练集/测试集 - 9/1）
"""
import os
import random
from shutil import copy, rmtree


def mk_dir(file_path):
    # 如果此文件夹存在
    if os.path.exists(file_path):
        # 删除原文件夹
        rmtree(file_path)
    # 重新创建此文件夹
    os.mkdir(file_path)


def main():
    # 随机种子，确保可复现
    random.seed(0)

    # 切分率 - 9/1
    split_rate = 0.1

    # 当前路径
    cwd = os.getcwd()
    data_root = cwd + "/data/flower_photos/"
    raw_photo_path = os.path.join(data_root, "raw_photos")
    assert os.path.exists(raw_photo_path), "path {} doesn't exist".format(raw_photo_path)

    flower_classes = [cls for cls in os.listdir(raw_photo_path) if os.path.isdir(os.path.join(raw_photo_path, cls))]
    print(flower_classes)

    # 建立训练集文件夹
    train_root = os.path.join(data_root, "train")
    mk_dir(train_root)
    for flower_class in flower_classes:
        mk_dir(os.path.join(train_root, flower_class))

    # 建立验证集文件夹
    val_root = os.path.join(data_root, "val")
    mk_dir(val_root)
    for flower_class in flower_classes:
        mk_dir(os.path.join(val_root, flower_class))

    for flower_class in flower_classes:
        cls_path = os.path.join(raw_photo_path, flower_class)
        imgs = os.listdir(cls_path)
        num = len(imgs)
        # 初始化验证集的索引
        eval_index = random.sample(imgs, k=int(num * split_rate))
        for i, img in enumerate(imgs):
            # 属于验证集
            if img in eval_index:
                # 将对应文件复制到相应目录
                from_image_path = os.path.join(cls_path, img)
                to_image_path = os.path.join(val_root, flower_class)
                copy(from_image_path, to_image_path)
            else:  # 属于训练集
                from_image_path = os.path.join(cls_path, img)
                to_image_path = os.path.join(train_root, flower_class)
                copy(from_image_path, to_image_path)
            print("\r[{}] processing [{}/{}]".format(flower_class, i + 1, num), end="")  # processing bar
        print()
    print("All done!!!")


if __name__ == '__main__':
    main()
