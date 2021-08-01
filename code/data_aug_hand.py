import os
import sys
import glob
from os.path import join
from multiprocessing import Pool

import cv2
import imageio
import numpy as np
import matplotlib.pyplot as plt
from skimage import io, color
from PIL import Image, ImageEnhance
import torch
from torch.utils.data import Dataset, DataLoader, TensorDataset
import torch.nn.functional as F
import torchvision.transforms as T

import joint_bilateral_filter as jbf
import config as c
import colour

# 版本说明：
    # 手工增强

offsets = (47.5, 2.4, 7.4)
scales = (25.6, 11.2, 16.8)
illuminant_RGB = np.array([0.31270, 0.32900])
illuminant_XYZ = np.array([0.34570, 0.35850])
chromatic_adaptation_transform = 'Bradford'
RGB_to_XYZ_matrix = np.array(
        [[0.41240000, 0.35760000, 0.18050000],
        [0.21260000, 0.71520000, 0.07220000],
        [0.01930000, 0.11920000, 0.95050000]])
XYZ_to_RGB_matrix = np.array(
            [[3.24062548, -1.53720797, -0.49862860],
            [-0.96893071, 1.87575606, 0.04151752],
            [0.05571012, -0.20402105, 1.05699594]])


def norm_lab_to_rgb(L, ab):
    ab = F.interpolate(ab, size=L.shape[2]) #, mode='bilinear')
    lab = torch.cat([L, ab], dim=1)
    print('norm_lab_to_rgb中，lab.shape', lab.shape)

    for i in range(3):
        lab[:, i] = lab[:, i] * scales[i] + offsets[i]
    # lab[:, 0].clamp_(0., 100.)
    # lab[:, 1:].clamp_(-128, 128)

    lab = [np.transpose(one_img, (1, 2, 0)) for one_img in lab.cpu().numpy()]
    xyz = colour.Lab_to_XYZ(lab)
    rgb = colour.XYZ_to_RGB(xyz, illuminant_XYZ, illuminant_RGB, XYZ_to_RGB_matrix,
                            chromatic_adaptation_transform)

    # rgb = [np.transpose(one_img, (2, 0, 1)) for one_img in rgb]
    return rgb


def rgb_img_to_Lab(im):
    # if im.shape[0] == 1:
    #     im = np.concatenate([im]*3, axis=0)
    # if im.shape[0] == 4:
    #     im = im[:3]
    # im = np.transpose(im, (1, 2, 0))

    xyz = colour.RGB_to_XYZ(im, illuminant_RGB, illuminant_XYZ,
                            RGB_to_XYZ_matrix, chromatic_adaptation_transform)
    lab = colour.XYZ_to_Lab(xyz).transpose(2, 0, 1)

    for i in range(3):
        lab[i] = (lab[i] - offsets[i]) / scales[i]
    return lab


class LabColorDataset(Dataset):
    def __init__(self, file_list):
        self.files = file_list
        self.to_tensor = T.ToTensor()

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        # PIL.Image.open读取的shape是这样的: (3, 256, 256)
        # im = Image.open(self.files[idx])
        # im = cv2.imread(self.files[idx], cv2.IMREAD_UNCHANGED) # .astype(np.float32)
        # print('检查dataloader读取顺序：', self.files[idx])
        im = imageio.imread(self.files[idx])
        try:
            lab = rgb_img_to_Lab(im)
            # 使得 噪声 服从 N(0,σ²) 正态分布, 大部分 落在  (-0.5, 0.5) 区间里.
            # mu, sigma = 0, 0.5 / 3
            # noise = np.clip(np.random.normal(mu, sigma, im.shape), -0.5, 0.4999)
            # 使得 噪声 服从 均匀分布 [low, high)
            noise = np.random.uniform(-0.5, 0.5, im.shape)
            im_augmented = im + noise
            lab_augmented = rgb_img_to_Lab(im_augmented)
            # lab_augmented = self.rgb_img_to_Lab(np.clip(im_augmented, 0,255))
            # 这一句，没有提高效果。2020-11-12
            return torch.Tensor(lab), torch.Tensor(lab_augmented)

        except:
            return self.__getitem__(idx+1)


# Data transforms for training and test/validation set
# transf = T.Compose([T.RandomHorizontalFlip(),
#                          T.RandomResizedCrop(c.img_dims_orig[0], scale=(0.2, 1.))])
# transf_test = T.Compose([T.Resize(c.img_dims_orig[0]),
#                          T.CenterCrop(c.img_dims_orig[0])])

if c.dataset == 'imagenet':
    with open('./imagenet/training_images.txt') as f:
        train_list = [join('./imagenet', fname[2:]) for fname in f.read().splitlines()]
    with open(c.validation_images) as f:
        test_list = [ t for t in f.read().splitlines()if t[0] != '#']
        test_list = [join('./imagenet', fname) for fname in test_list]
        if c.val_start is not None:
            test_list = test_list[c.val_start:c.val_stop]
else:
    train_dir = '../dataset/coco17/train'
    test_dir = '../dataset/coco17/test'
    train_list = sorted(glob.glob(join(train_dir, '*.jpg')))
    test_list = sorted(glob.glob(join(test_dir, '*.jpg')))


train_data = LabColorDataset(train_list)
test_data = LabColorDataset(test_list)

train_loader = DataLoader(train_data, batch_size=c.batch_size, shuffle=True, num_workers=8, pin_memory=True, drop_last=True)
# test_loader = DataLoader(test_data,  batch_size=min(64, len(test_list)), shuffle=c.shuffle_val, num_workers=4, pin_memory=True, drop_last=False)
test_loader = DataLoader(test_data,  batch_size=c.batch_size, shuffle=c.shuffle_val, num_workers=0, pin_memory=True, drop_last=False)

