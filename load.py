# -*- coding:utf-8 -*-
from __future__ import print_function, division
from scipy.io import loadmat as load
import matplotlib.pyplot as plt
import numpy as np


def reformat(samples, labels):
    # 改变原始数据的形状
    #      0       1        2        3             3       0        1        2
    # （图片高， 图片宽， 通道数， 图片数） -> （图片数，图片高， 图片宽， 通道数）
    # labels 变成 one-hot encoding, eg: [1] -> [0, 1, 0, 0, 0, 0, 0, 0, 0, 0]
    # [2] -> [0, 0, 1, 0, 0, 0, 0, 0, 0, 0]   [10] -> [1, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    samples = np.transpose(samples, (3, 0, 1, 2))

    labels = np.array([x[0] for x in labels])
    one_hot_labels = []
    for num in labels:
        one_hot = [0.0] *10
        if num == 10:
            one_hot[0] = 1.0
        else:
            one_hot[num] = 1.0
        one_hot_labels.append(one_hot)
    labels = np.array(one_hot_labels).astype(np.float32)
    return samples, labels


def normalize(samples):
    # 灰度化：从三色通道 -> 单色通道， 目的：省内存，加快训练速度
    # (R + G + B)/3
    # 将图片从0 ～ 255 线性映射到 -1.0 ～ +1.0
    # shape （图片数，图片高， 图片宽， 通道数）先去掉一个通道数，然后在把前三个相加
    samples = np.add.reduce(samples, keepdims=True, axis=3)
    samples = samples/3.0
    samples = samples / 128 - 1
    return samples


def distributioin(labels, name):
    # 查看一下每个labels的分布。 就是比例。 再画个统计图
    # 0 - 9有多少个

    count = {}
    for label in labels:
        key = 0 if label[0] == 10 else label[0]
        if key in count:
            count[key] += 1
        else:
            count[key] = 1
    x = []
    y = []
    for k, v in count.items():
        # print(k, v)
        x.append(k)
        y.append(v)

    y_pos = np.arange(len(x))
    plt.bar(y_pos, y, align='center', alpha=0.5)
    plt.xticks(y_pos, x)
    plt.ylabel('Count')
    plt.title(name + 'Label Distribution')
    plt.show()


def inspect(dataset, labels, i):
    # 显示图片看看
    print(labels[i])
    plt.imshow(dataset[i])
    plt.show()


traindata = load('../data/train_32x32.mat')
testdata = load('../data/test_32x32.mat')
extradata = load('../data/extra_32x32.mat')


print('traindata shape:', traindata['X'].shape)
print('traindata shape:', traindata['y'].shape)
print()
print('testdata shape:', testdata['X'].shape)
print('testdata shape:', testdata['y'].shape)
print()
print('extradata shape:', extradata['X'].shape)
print('extradata shape:', extradata['y'].shape)
print()

train_samples = traindata['X']
train_labels  = traindata['y']
test_samples = testdata['X']
test_labels  = testdata['y']
extra_samples = extradata['X']
extra_labels  = extradata['y']

n_train_samples, _train_labels = reformat(train_samples, train_labels)
n_test_samples, _test_labels = reformat(test_samples, test_labels)
n_extra_samples, _extra_labels = reformat(extra_samples, extra_labels)

_train_samples = normalize(n_train_samples)
_test_samples = normalize(n_test_samples)
_extra_samples = normalize(n_extra_samples)

num_labels = 10
image_size = 32
num_channels = 1

if __name__ == '__main__':
    # 探索数据
    # inspect(_train_samples, _train_samples, 1234)
    distributioin(train_labels, 'Train Labels')
    distributioin(test_labels, 'Test Labels')
