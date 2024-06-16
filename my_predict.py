import torch
import torchvision
from matplotlib import pyplot as plt
from torchvision.transforms.functional import crop
from d2l.torch import d2l
import  os
from some_define import ResNetUNet

def read_voc_images(voc_dir, is_train=True):
    """读取所有VOC图像并标注,返回features, labels"""
    txt_fname = os.path.join(r'VOC2012', 'ImageSets', 'Segmentation',
                             'train.txt' if is_train else 'val.txt')

    mode = torchvision.io.image.ImageReadMode.RGB

    with open(txt_fname, 'r') as f:
        images = f.read().split()

    features, labels = [], []

    for i, fname in enumerate(images):
        features.append(torchvision.io.read_image(os.path.join(
            voc_dir, 'JPEGImages', f'{fname}.jpg')))

        labels.append(torchvision.io.read_image(os.path.join(
            voc_dir, 'SegmentationClass', f'{fname}.png'), mode))

    return features, labels

# load_net
net = torch.load("net_30_lowlr10.pth")

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
plt.savefig(r'out.png')
plt.show()