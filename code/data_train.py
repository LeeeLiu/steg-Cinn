import os
import sys
import glob
from os.path import join
from multiprocessing import Pool

import cv2
import h5py
os.environ["HDF5_USE_FILE_LOCKING"] = 'FALSE'

# 版本说明
    # 11-23_data_train.py
        # 和 data_09-21.py 区别是
        # 数据集目录，不同
        # 数据集是合成的：PNG(取整TIFF) + label Z(.h5格式)

import imageio
import numpy as np
import matplotlib.pyplot as plt
from skimage import io, color
from PIL import Image, ImageEnhance
import torch
from torch.utils.data import Dataset, DataLoader, TensorDataset
import torch.nn.functional as F
import torchvision.transforms as T
from tqdm import tqdm

import joint_bilateral_filter as jbf
import config_train as c
import colour


offsets = (47.5, 2.4, 7.4)
scales = (25.6, 11.2, 16.8)

def apply_filt(args):
    '''multiprocessing wrapper for applying the joint bilateral filter'''
    L_i, ab_i = args
    return jbf.upsample(L_i[0], ab_i, s_x=6, s_l=0.10)


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
    def __init__(self, file_list, label_list):
        self.files = file_list
        self.labels = label_list
        self.to_tensor = T.ToTensor()

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        # PIL.Image.open读取的shape是这样的: (3, 256, 256)
        # im = Image.open(self.files[idx])
        # im = cv2.imread(self.files[idx], cv2.IMREAD_UNCHANGED) # .astype(np.float32)

        # print('检查Tiff目录：', self.files[idx])
        # print('检查Z目录：', self.labels[idx])

        # 读取PNG（取整之后的 Tiff）
        im = imageio.imread(self.files[idx])
        # 读取标签（Z）
        f = h5py.File(self.labels[idx], 'r')  # 打开h5文件
        label_Z = []
        for key in f.keys():
            # print(f[key].name)
            # print(f[key].shape)
            # print(f[key].value)
            label_Z.append( torch.Tensor(np.array(f[key][()])).cuda() )
        f.close()

        try:
            lab = rgb_img_to_Lab(im)
            return torch.Tensor(lab), label_Z

        except:
            return self.__getitem__(idx+1)

#
# transf = T.Compose([T.RandomHorizontalFlip(),
#                          T.RandomResizedCrop(c.img_dims_orig[0], scale=(0.2, 1.))])
# transf_test = T.Compose([T.Resize(c.img_dims_orig[0]),
#                          T.CenterCrop(c.img_dims_orig[0])])

def get_train_loader(id_cINN_pre):
    # todo
    train_dir = '../dataset/gen_cover_Z_by_cINN_{}/train'.format(id_cINN_pre)
    # test_dir = '../dataset/gen_cover_Z_by_cINN_{}/test'.format(id_cINN_pre)

    train_list = sorted(glob.glob(join(train_dir, '*.png')))
    train_label_list = sorted(glob.glob(join(train_dir, '*.h5')))

    # test_list = sorted(glob.glob(join(test_dir, '*.png')))
    # test_label_list = sorted(glob.glob(join(test_dir, '*.h5')))

    train_data = LabColorDataset(train_list, train_label_list)
    # test_data = LabColorDataset(test_list, test_label_list)

    train_loader = DataLoader(train_data, batch_size=c.batch_size, shuffle=c.shuffle_train, num_workers=0, pin_memory=False, drop_last=True)
    # test_loader = DataLoader(test_data,  batch_size=c.batch_size, shuffle=c.shuffle_val, num_workers=0, pin_memory=True, drop_last=False)

    return train_loader


# if __name__ == '__main__':
