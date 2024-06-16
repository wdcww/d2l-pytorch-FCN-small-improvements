# ####################################################################
# 去看some_define.py吧: lr = 0.001, wd=0.001, num_epochs = 30,
#  optim.SGD , cross_entropy ,
# 使用正态分布的Xavier初始化1×1卷积层'final_conv'
# 使用双线性插值的上采样初始化转置卷积层'transpose_conv'
# #####################################################################

from some_define import read_voc_images, bilinear_kernel, train_batch_ch13, VOCSegDataset, crop_size, batch_size, lr, \
    wd, num_epochs, devices

import numpy as np
import torch
import torchvision
from d2l import torch as d2l
from matplotlib import pyplot as plt
from torch import nn
from torch.utils.data import Dataset
from torch.nn import functional as F


# # 1
# # # # 展示一下部分训练集
# train_features, train_labels = read_voc_images(r'VOC2012', True)
# n = 5
# imgs = train_features[0:n] + train_labels[0:n]   # train_features 是 JPEGImages文件中的图
#                                                  # train_labels 是 SegmentationClass 中的图
# imgs = [img.permute(1,2,0) for img in imgs]
# d2l.show_images(imgs, 2, n)
# plt.show()
# # # #

# # 2
# # # 全卷积神经网络 ############################################
pretrained_net = torchvision.models.resnet18(weights=None)
local_weights_path = r'resnet18-f37072fd.pth'
state_dict = torch.load(local_weights_path)
pretrained_net.load_state_dict(state_dict)

net = nn.Sequential(*list(pretrained_net.children())[:-2])
net.add_module('final_conv', nn.Conv2d(512, 21, kernel_size=1))
net.add_module('transpose_conv', nn.ConvTranspose2d(21, 21, kernel_size=64, padding=16, stride=32))


# # 初始化转置卷积层
W = bilinear_kernel(21, 21, 64)
net.transpose_conv.weight.data.copy_(W)  # 用双线性插值的上采样初始化转置卷积层

# # 初始化1×1卷积层
import torch.nn.init as init
init.xavier_normal_(net.final_conv.weight) # 使用正态分布进行Xavier初始化

# print(net)
# # ###############################################################

# # 3
# # train ==============================================================

# # 定义三个全局变量list,用于一会的loss可视化
acc_tra = []
acc_val = []
train_losses = []


def train_ch13(net, train_iter, test_iter, loss, trainer, num_epochs,
               devices=d2l.try_all_gpus()):
    """Train a model with mutiple GPUs (defined in Chapter 13).

    Defined in :numref:`sec_image_augmentation`"""
    timer, num_batches = d2l.Timer(), len(train_iter)

    net = nn.DataParallel(net, device_ids=devices).to(devices[0])
    for epoch in range(num_epochs):
        # Sum of training loss, sum of training accuracy, no. of examples,
        # no. of predictions
        metric = d2l.Accumulator(4)
        for i, (features, labels) in enumerate(train_iter):
            timer.start()
            l, acc = train_batch_ch13(net, features, labels, loss, trainer, devices)
            metric.add(l, acc, labels.shape[0], labels.numel())
            timer.stop()

        test_acc = d2l.evaluate_accuracy_gpu(net, test_iter)

        print(
            f'loss {metric[0] / metric[2]:.3f},' f'train acc 'f'{metric[1] / metric[3]:.3f}, 'f'test acc {test_acc:.3f}')

        acc_tra.append(metric[1] / metric[3])
        acc_val.append(test_acc)
        train_losses.append(metric[0] / metric[2])

        print(f'{metric[2] * num_epochs / timer.sum():.1f} examples/sec on 'f'{str(devices)}')
    # # 最后保存一下模型
    torch.save(net, "origin_net_{}.pth".format(epoch + 1))
    print("模型已保存")


# # DataLoader
train_iter = torch.utils.data.DataLoader(
        VOCSegDataset(True, crop_size, r'VOC2012'), batch_size,shuffle=True, drop_last=True)

test_iter = torch.utils.data.DataLoader(
        VOCSegDataset(False, crop_size, r'VOC2012'), batch_size,drop_last=True)


# loss
def loss(inputs, targets):
    return F.cross_entropy(inputs, targets, reduction='none').mean(1).mean(1)

# optim
trainer = torch.optim.SGD(net.parameters(), lr=lr, weight_decay=wd)

# 调用训练函数
train_ch13(net, train_iter, test_iter, loss, trainer, num_epochs, devices)

# # =========================================================================

# # 4
# # # 可视化------------------------------------------------------------------
fig = plt.figure()
# 添加第一个子图，设置标题和坐标轴标签
ax1 = fig.add_subplot(1, 2, 1)
ax1.plot(list(range(1, len(acc_tra) + 1)), np.array(acc_tra), color='red', marker='o', label='acc_tra')
ax1.plot(list(range(1, len(acc_val) + 1)), np.array(acc_val), color='black', marker='o', label='acc_val')
plt.legend()

# 添加第二个子图，设置标题和坐标轴标签
ax2 = fig.add_subplot(1, 2, 2)
ax2.plot(list(range(1, len(train_losses) + 1)), train_losses, color='blue', marker='o', label='train_loss')
plt.legend()

# 保存图片
plt.savefig(r"origin_loss.png")
plt.show()
plt.close()

# #-----------------------------------------------------------------------------
