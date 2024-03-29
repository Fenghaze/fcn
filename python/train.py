# -*- coding: utf-8 -*-

from __future__ import print_function

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable
from torch.utils.data import DataLoader

from fcn import VGGNet, FCN32s, FCN16s, FCN8s, FCNs
from Cityscapes_loader import CityScapesDataset
from CamVid_loader import CamVidDataset

from matplotlib import pyplot as plt
import numpy as np
import time
import sys
import os


#n_class    = 20
n_class    = 32
batch_size = 1
epochs     = 500
lr         = 1e-4
momentum   = 0
w_decay    = 1e-5
step_size  = 50
gamma      = 0.5
configs    = "FCNs-BCEWithLogits_batch{}_epoch{}_RMSprop_scheduler-step{}-gamma{}_lr{}_momentum{}_w_decay{}".format(batch_size, epochs, step_size, gamma, lr, momentum, w_decay)
print("Configs:", configs)

# if sys.argv[1] == 'CamVid':
#     root_dir   = "CamVid/"
# else:
#     root_dir   = "CityScapes/"
root_dir = '../../../datasets/CamVid'
train_file = os.path.join(root_dir, "train.csv")
val_file   = os.path.join(root_dir, "val.csv")

# create dir for model
model_dir = "models"
if not os.path.exists(model_dir):
    os.makedirs(model_dir)
model_path = os.path.join(model_dir, configs)

use_gpu = torch.cuda.is_available()
num_gpu = list(range(torch.cuda.device_count()))

# 实例化数据集
train_data = CamVidDataset(csv_file=train_file, phase='train')
val_data = CamVidDataset(csv_file=val_file, phase='val', flip_rate=0)
# 加载数据集，返回可迭代对象
train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=0)
val_loader = DataLoader(val_data, batch_size=batch_size, num_workers=2)
# 微调模型：VGGNet去掉全连接层
vgg_model = VGGNet(requires_grad=True, remove_fc=True)
fcn_model = FCNs(pretrained_net=vgg_model, n_class=n_class)

if use_gpu:
    ts = time.time()
    vgg_model = vgg_model.cuda()
    fcn_model = fcn_model.cuda()
    fcn_model = nn.DataParallel(fcn_model, device_ids=num_gpu)
    print("Finish cuda loading, time elapsed {}".format(time.time() - ts))

# 定义损失函数
criterion = nn.BCEWithLogitsLoss()
# 定义梯度下降算法
optimizer = optim.RMSprop(fcn_model.parameters(), lr=lr, momentum=momentum, weight_decay=w_decay)
# 学习率衰减器
scheduler = lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)  # decay LR by a factor of 0.5 every 30 epochs

# create dir for score
score_dir = os.path.join("scores", configs)
if not os.path.exists(score_dir):
    os.makedirs(score_dir)

# 评价标准
IU_scores    = np.zeros((epochs, n_class))
pixel_scores = np.zeros(epochs)


def train():
    for epoch in range(epochs):
        scheduler.step()

        ts = time.time()
        for iter, batch in enumerate(train_loader):
            optimizer.zero_grad()

            if use_gpu:
                inputs = Variable(batch['X'].cuda())
                labels = Variable(batch['Y'].cuda())
            else:
                inputs, labels = Variable(batch['X']), Variable(batch['Y'])
            # outputs.shape = [n_class,h,w]，通道c的每个像素对应类别c的预测概率， n_class=32， c=0,1,...,n_class-1
            outputs = fcn_model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            if iter % 10 == 0:
                print("epoch{}, iter{}, loss: {}".format(epoch, iter, loss.item()))
        
        print("Finish epoch {}, time elapsed {}".format(epoch, time.time() - ts))
        torch.save(fcn_model, model_path)

        val(epoch)


def val(epoch):
    fcn_model.eval()
    total_ious = []
    pixel_accs = []
    for iter, batch in enumerate(val_loader):
        if use_gpu:
            inputs = Variable(batch['X'].cuda())
        else:
            inputs = Variable(batch['X'])

        output = fcn_model(inputs)
        output = output.data.cpu().numpy()

        N, _, h, w = output.shape
        """
        图像大小为 h * w，故总的像素标签大小为 n_class * h * w
        n_class * 1 * 1 为一个像素标签，表示为[0,0,0,...,1,...,0]，假设标签索引值为 15，即这个像素是属于第15个类别
        假设output中的一个像素分类预测概率为  [0.02,0,...,0.8,...,0.03]，找到最大索引值为 15，那么说明这个像素分类正确
        索引值就是类别的标签值，计算像素的分类准确率，就是计算pred和target中索引值相等的个数
        # （N,32,480,640）→（N,480,640,32）→（480*640,32）→（N,480,640），取每行概率最大值的索引，再重塑成 N,H,W
        pred=(H,W)          target=（H,W）
        4 4 7 7  7          4 4 4 7  7
        4 4 4 7  10         4 4 7 7  7
        4 4 4 10 10         4 4 4 10 10
        """
        pred = output.transpose(0, 2, 3, 1).reshape(-1, n_class).argmax(axis=1).reshape(N, h, w)
        target = batch['l'].cpu().numpy().reshape(N, h, w)
        for p, t in zip(pred, target):
            total_ious.append(iou(p, t))
            pixel_accs.append(pixel_acc(p, t))

    # Calculate average IoU
    total_ious = np.array(total_ious).T  # n_class * val_len
    ious = np.nanmean(total_ious, axis=1)
    pixel_accs = np.array(pixel_accs).mean()
    print("epoch{}, pix_acc: {}, meanIoU: {}, IoUs: {}".format(epoch, pixel_accs, np.nanmean(ious), ious))
    IU_scores[epoch] = ious
    np.save(os.path.join(score_dir, "meanIU"), IU_scores)
    pixel_scores[epoch] = pixel_accs
    np.save(os.path.join(score_dir, "meanPixel"), pixel_scores)


# borrow functions and modify it from https://github.com/Kaixhin/FCN-semantic-segmentation/blob/master/main.py
# Calculates class intersections over unions
def iou(pred, target):
    ious = []
    for cls in range(n_class):
        pred_inds = pred == cls
        target_inds = target == cls
        intersection = pred_inds[target_inds].sum()
        union = pred_inds.sum() + target_inds.sum() - intersection
        if union == 0:
            ious.append(float('nan'))  # if there is no ground truth, do not include in evaluation
        else:
            ious.append(float(intersection) / max(union, 1))
        # print("cls", cls, pred_inds.sum(), target_inds.sum(), intersection, float(intersection) / max(union, 1))
    return ious


def pixel_acc(pred, target):
    correct = (pred == target).sum()
    total   = (target == target).sum()
    return correct / total


if __name__ == "__main__":
    #val(0)  # show the accuracy before training
    train()
