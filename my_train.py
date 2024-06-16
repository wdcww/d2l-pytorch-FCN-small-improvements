# ####################################################################
# lr = 1e-4, wd=0.001, num_epochs = 30,
#  optim.Adam , cross_entropy ,
#  改变了net 、优化器(SGD-->Adam)、lr = 1e-4
# #####################################################################

from some_define import train_batch_ch13, VOCSegDataset, crop_size, batch_size, \
    wd, num_epochs, devices, ResNetUNet

import numpy as np
import torch
from d2l import torch as d2l
from matplotlib import pyplot as plt
from torch import nn
from torch.utils.data import Dataset
from torch.nn import functional as F


net = ResNetUNet(21)
print(net)


# # train ==============================================================

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
    torch.save(net, "net_{}.pth".format(epoch + 1))
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
trainer = torch.optim.Adam(net.parameters(), lr = 1e-4, weight_decay=wd)

# 调用训练函数
train_ch13(net, train_iter, test_iter, loss, trainer, num_epochs, devices)

# # =========================================================================


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
plt.savefig(r"loss.png")
plt.close()

# #-----------------------------------------------------------------------------
