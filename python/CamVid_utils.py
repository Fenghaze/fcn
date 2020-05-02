# -*- coding: utf-8 -*-

from __future__ import print_function

from matplotlib import pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import scipy.misc
import imageio
import random
import os


#############################
    # global variables #
#############################
# root_dir          = "CamVid/"
root_dir = "../../../datasets/CamVid"
data_dir          = os.path.join(root_dir, "701_StillsRaw_full")    # train data
label_dir         = os.path.join(root_dir, "LabeledApproved_full")  # train label
label_colors_file = os.path.join(root_dir, "label_colors.txt")      # color to label

val_label_file    = os.path.join(root_dir, "val.csv")               # validation file
train_label_file  = os.path.join(root_dir, "train.csv")             # train file

# create dir for label index
label_idx_dir = os.path.join(root_dir, "Labeled_idx")
if not os.path.exists(label_idx_dir):
    os.makedirs(label_idx_dir)

# 关于GT的字典
label2color = {}
color2label = {}
label2index = {}
index2label = {}

# 数据集划分（训练集：验证集 = 9:1）
def divide_train_val(val_rate=0.1, shuffle=True, random_seed=None):
    data_list   = os.listdir(data_dir)
    data_len    = len(data_list)
    val_len     = int(data_len * val_rate)

    if random_seed:
        random.seed(random_seed)

    # 在一个范围内随机取样，取样个数为 data_len（相当于打乱数字）
    if shuffle:
        data_idx = random.sample(range(data_len), data_len)
    else:
        data_idx = list(range(data_len))

    # 分别为验证集合训练集分配图片
    val_idx     = [data_list[i] for i in data_idx[:val_len]]
    train_idx   = [data_list[i] for i in data_idx[val_len:]]

    # create val.csv
    v = open(val_label_file, "w")
    v.write("img,label\n")
    for idx, name in enumerate(val_idx):
        # 排除缺失值
        if 'png' not in name:
            continue
        img_name = os.path.join(data_dir, name)
        lab_name = os.path.join(label_idx_dir, name)
     #   lab_name = lab_name.split(".")[0] + "_L.png.npy"
        lab_name = "../../.." + lab_name.split(".")[-2] + "_L.png.npy"
        v.write("{},{}\n".format(img_name, lab_name))

    # create train.csv
    t = open(train_label_file, "w")
    t.write("img,label\n")
    for idx, name in enumerate(train_idx):
        if 'png' not in name:
            continue
        img_name = os.path.join(data_dir, name)
        lab_name = os.path.join(label_idx_dir, name)
       # lab_name = lab_name.split(".")[0] + "_L.png.npy"
        lab_name = "../../.." + lab_name.split(".")[-2] + "_L.png.npy"
        t.write("{},{}\n".format(img_name, lab_name))

# 处理类别标签（图像中存在多个类别，如车、动物、树木等，每个类别标记为不同的RGB）
def parse_label():
    # change label to class index
    f = open(label_colors_file, "r").read().split("\n")[:-1]  # ignore the last empty line
    for idx, line in enumerate(f):
        # 获取类别名（如：Animal、Tree等）
        label = line.split()[-1]
        # 类别对应的RGB
        color = tuple([int(x) for x in line.split()[:-1]])
        print(label, color)
        # RGB-类别字典
        label2color[label] = color
        color2label[color] = label
        # 标签-类别字典
        label2index[label] = idx # {'Animal':0, ..., 'Tree':10, ...}
        index2label[idx]   = label # {0:'Animal', ..., 10:'Tree', ...}
        # rgb = np.zeros((255, 255, 3), dtype=np.uint8)
        # rgb[..., 0] = color[0]
        # rgb[..., 1] = color[1]
        # rgb[..., 2] = color[2]
        # imshow(rgb, title=label)
    
    for idx, name in enumerate(os.listdir(label_dir)):
        filename = os.path.join(label_idx_dir, name)
        # 如果存在打好标记的图片，则跳过不用处理
        if os.path.exists(filename + '.npy'):
            print("Skip %s" % (name))
            continue
        # 开始为每个训练图片的像素打标
        print("Parse %s" % (name))
        img = os.path.join(label_dir, name)
        # img = scipy.misc.imread(img, mode='RGB')
        # 读取图片
        img = imageio.imread(img)
        height, weight, _ = img.shape
        # 创建一个零矩阵，该矩阵保存了图片的像素标签
        idx_mat = np.zeros((height, weight))

        # 为图像中的每个像素点打上类别标签
        for h in range(height):
            for w in range(weight):
                color = tuple(img[h, w])
                try:
                    # 通过颜色获取类别
                    label = color2label[color]
                    # 通过类别获取类别标签
                    index = label2index[label]
                    # 给像素点打标签
                    idx_mat[h, w] = index
                except:
                    print("error: img:%s, h:%d, w:%d" % (name, h, w))
        # 将矩阵转换为二进制数据
        idx_mat = idx_mat.astype(np.uint8)
        # 保存打好像素标签的图片，这是二进制文件（.npy）
        np.save(filename, idx_mat)
        print("Finish %s" % (name))

    # test some pixels' label    
    img = os.path.join(label_dir, os.listdir(label_dir)[0])
    # img = scipy.misc.imread(img, mode='RGB')
    img = imageio.imread(img)
    test_cases = [(555, 405), (0, 0), (380, 645), (577, 943)]
    test_ans   = ['Car', 'Building', 'Truck_Bus', 'Car']
    for idx, t in enumerate(test_cases):
        color = img[t]
        assert color2label[tuple(color)] == test_ans[idx]


'''debug function：查看图片'''
def imshow(img, title=None):
    try:
        img = mpimg.imread(img)
        imgplot = plt.imshow(img)
    except:
        plt.imshow(img, interpolation='nearest')

    if title is not None:
        plt.title(title)
    
    plt.show()


if __name__ == '__main__':
    divide_train_val(random_seed=1)
    parse_label()
