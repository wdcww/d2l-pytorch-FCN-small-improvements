from some_define import read_voc_images

import torch
import torchvision
from matplotlib import pyplot as plt
from torchvision.transforms.functional import crop
from d2l.torch import d2l

# load_net
net = torch.load("origin_net_30.pth")

devices=d2l.try_all_gpus()
transform = torchvision.transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
def predict(img):
    img = transform(img.float() / 255).unsqueeze(0)
    pred = net(img.to(devices[0])).argmax(dim=1)
    return pred.reshape(pred.shape[1], pred.shape[2])

def label2image(pred):
    colormap = torch.tensor(d2l.VOC_COLORMAP, device=devices[0])
    X = pred.long()
    return colormap[X, :]

test_images, test_labels = read_voc_images(r'VOC2012', False)  # 这个只是把图片读出来

n, imgs = 4, []
for i in range(n):
    crop_rect = (0, 0, 320, 480)
    X = torchvision.transforms.functional.crop(test_images[i], *crop_rect)  # 每次,第i张图片crop
    pred = label2image(predict(X))
    imgs += [X.permute(1,2,0),
             torchvision.transforms.functional.crop(test_labels[i], *crop_rect).permute(1, 2, 0),
             pred.cpu()]

d2l.show_images(imgs[::3] + imgs[1::3] + imgs[2::3], 3, n, scale=2)
plt.savefig(r'origin_out.png')
plt.show()