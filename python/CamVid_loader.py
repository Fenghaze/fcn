# -*- coding: utf-8 -*-

from __future__ import print_function

import imageio
from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
import scipy.misc
import random
import os

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import utils


root_dir   = "../../../datasets/CamVid"
train_file = os.path.join(root_dir, "train.csv")
val_file   = os.path.join(root_dir, "val.csv")

num_class = 32
means     = np.array([103.939, 116.779, 123.68]) / 255. # mean of three channels in the order of BGR
h, w      = 720, 960
train_h   = int(h * 2 / 3)  # 480
train_w   = int(w * 2 / 3)  # 640
val_h     = int(h/32) * 32  # 704
val_w     = w               # 960

# 定义数据集
class CamVidDataset(Dataset):
    def __init__(self, csv_file, phase, n_class=num_class, crop=True, flip_rate=0.5):
        """
        :param csv_file: csv文件，保存了原图和 GT 的数据
        :param phase: train/val
        :param n_class:
        :param crop: 是否裁剪
        :param flip_rate:
        """
        # data是一个DataFrame数据
        self.data      = pd.read_csv(csv_file)
        self.means     = means
        self.n_class   = n_class

        self.flip_rate = flip_rate
        self.crop      = crop
        # 如果是测试集，将图片设置为新的高度
        if phase == 'train':
            self.new_h = train_h
            self.new_w = train_w
        # 如果是验证集，不对图片进行预处理
        elif phase == 'val':
            self.flip_rate = 0.
            self.crop = False
            self.new_h = val_h
            self.new_w = val_w


    def __len__(self):
        return len(self.data)

    # 根据 idx 返回一个样本
    def __getitem__(self, idx):
        # 获取DataFrame的数据：获取索引值为idx，第1列的数据（原图路径）
        img_name   = self.data.iloc[idx, 0]
        # 获取图片对象
        img = imageio.imread(img_name)
        # img        = scipy.misc.imread(img_name, mode='RGB')
        # 获取索引值为idx，第2列的数据（像素标签路径）
        label_name = self.data.iloc[idx, 1]
        # np.load：读取二进制文件
        label      = np.load(label_name)

        # 裁剪图像和标签
        if self.crop:
            h, w, _ = img.shape
            top   = random.randint(0, h - self.new_h)
            left  = random.randint(0, w - self.new_w)
            img   = img[top:top + self.new_h, left:left + self.new_w]
            label = label[top:top + self.new_h, left:left + self.new_w]

        if random.random() < self.flip_rate:
            img   = np.fliplr(img)
            label = np.fliplr(label)

        # reduce mean
        img = img[:, :, ::-1]  # switch to BGR
        img = np.transpose(img, (2, 0, 1)) / 255.
        img[0] -= self.means[0]
        img[1] -= self.means[1]
        img[2] -= self.means[2]

        # convert to tensor
        img = torch.from_numpy(img.copy()).float()
        label = torch.from_numpy(label.copy()).long()

        # create one-hot encoding：这是什么意思，为什么这个32通道数的张量是target
        h, w = label.size()
        # 创建一个零张量，长宽与label一致
        target = torch.zeros(self.n_class, h, w)
        for c in range(self.n_class):
            target[c][label == c] = 1

        sample = {'X': img, 'Y': target, 'l': label}

        return sample

# 展示batch图片
def show_batch(batch):
    img_batch = batch['X']
    img_batch[:,0,...].add_(means[0])
    img_batch[:,1,...].add_(means[1])
    img_batch[:,2,...].add_(means[2])
    batch_size = len(img_batch)

    grid = utils.make_grid(img_batch)
    plt.imshow(grid.numpy()[::-1].transpose((1, 2, 0)))

    plt.title('Batch from dataloader')


if __name__ == "__main__":
    # 实例化数据集
    train_data = CamVidDataset(csv_file=train_file, phase='train')

    # show a batch
    batch_size = 4
    for i in range(batch_size):
        sample = train_data[i]
        print(i, sample['X'].size(), sample['Y'].size())

    # 加载数据集，返回一个可迭代对象
    dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=0)

    for i, data in enumerate(dataloader):
        print(i, data['X'].size(), data['Y'].size())
    
        # observe 4th data
        if i == 3:
            plt.figure()
            show_batch(data)
            plt.axis('off')
            plt.ioff()
            plt.show()
            break
