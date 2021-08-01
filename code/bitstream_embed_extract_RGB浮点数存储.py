
import glob
import sys
from os.path import join
import os
from torch.utils.data import Dataset, DataLoader, TensorDataset
import imageio
import torch
import torch.nn as nn
import numpy as np
import matplotlib

import viz
from mapping import bit2dec, dec2bit, compare_bitstream

matplotlib.use('Agg')
import config as c

if c.no_cond_net:
    import model_no_cond as model
else:
    import model
import data

def cal_avg_var(z):
    avg0 = np.average(np.array(z[0].cpu()))
    avg1 = np.average(np.array(z[1].cpu()))
    avg2 = np.average(np.array(z[2].cpu()))
    avg3 = np.average(np.array(z[3].cpu()))
    avg = (avg0 + avg1 + avg2 + avg3) / 4
    var = np.var(np.array(z[0].cpu()))
    return avg, var


# 嵌入
def sample_z(N, outputs, args):
    if args.mapping=='True':
        print('i am mapping')
        bitstream = ""  # 一维01字符串
        map_sampled_z = []  # 保持和outputs一样的维度（四部分组成）
        for o in outputs:
            shape = list(o.shape)
            shape[0] = N
            bits, z = bit2dec(shape, size_group=args.size_k, boundary=args.delta, gap=args.g / 2)
            bitstream += bits
            map_sampled_z.append(torch.tensor(z, dtype=torch.float32).cuda())
        return bitstream, map_sampled_z

    if args.mapping=='False':
        sampled_z = []
        for o in outputs:
            shape = list(o.shape)
            shape[0] = N
            sampled_z.append(torch.randn(shape).cuda())
        return "", sampled_z

# 提取
def extract(multi_dim_latent, args):
    bitstream = ""  # 一维01字符串
    if args.mapping:
        for noise in multi_dim_latent:
            noise = noise.squeeze().cpu().numpy()
            bits = dec2bit(noise, args.size_k, args.g/2)
            bitstream += bits
    return bitstream


if __name__ == '__main__':
    model_name = c.filename
    model.load(model_name)
    model.combined_model.eval()
    model.combined_model.module.inn.eval()

    if not c.no_cond_net:
        model.combined_model.module.feature_network.eval()
        model.combined_model.module.fc_cond_network.eval()

    ## begin
    import argparse
    parser = argparse.ArgumentParser(description='cINN-based image steganography')
    parser.add_argument('--mapping', type=str, help='map means embed msg')
    parser.add_argument('--size_k', type=int, help='k means size group(capacity)')
    parser.add_argument('--delta', type=float, help='delta means boundary')
    parser.add_argument('--g', type=float, help='g/2 means gap')
    args = parser.parse_args()

    # 数据X是Lab格式
    for x in data.test_loader:
        with torch.no_grad():
            n = x.shape[0]
            ims = []
            x = x[:n]
            print('-------'*10)
            # 反向 嵌入
            # 为了采样，先预测Z形状
            xl, xab, xcnd, _ = model.prepare_batch(x)
            output_z = model.cinn(xab, xcnd)
            # 按照output_z的形状，生成N个样本
            torch.manual_seed(c.seed)
            ## 嵌入比特流
            bitstream1, z = sample_z(N=n, outputs=output_z, args=args)

            x_l, x_ab, cond, ab_pred = model.prepare_batch(x)
            x_ab_sampled = model.combined_model.module.reverse_sample(list(np.asarray(z)), cond)
            print('(1)x_l, x_ab_sampled 是不是浮点数', x_l.dtype, x_ab_sampled.dtype)

            image = data.norm_lab_to_rgb(x_l, x_ab_sampled)
            print('反向 嵌入：')  # 生成的image shape is {}:'.format(image.shape))
            stego_dir = './output/stego'
            if not os.path.exists(stego_dir):
                os.makedirs(stego_dir)
            import cv2
            print('一，保存RGB之前，确认是不是浮点数', image.dtype)
            for i in range(image.shape[0]):
                cv2.imwrite(stego_dir+'/{}.exr'.format(i), image[i].squeeze().transpose(1, 2, 0))

            print('-------' * 10)

            X_lab = data.batch_RGB_to_Lab(stego_dir)
            print('(2)从stego_loader读取的lab  : ',X_lab.dtype, X_lab.shape)
            L, ab, cnd, _ = model.prepare_batch(X_lab)
            print('condition[0]的误差是：', np.average(abs(np.array(cond[0].cpu()) - np.array(cnd[0].cpu()))))
            print('灰度L的误差是：', np.average(abs(np.array(x_l.cpu()) - np.array(L.cpu()))))
            print('色度x_ab的误差是：', np.average(abs(np.array(x_ab_sampled.cpu()) - np.array(ab.cpu()))))
            outputs = model.cinn(ab, cnd)

            bitstream2 = extract(outputs, args)
            compare_bitstream(bitstream1, bitstream2, args)

            dif0 = np.sum(abs(np.array(z[0].cpu()) - np.array(outputs[0].cpu())))
            dif1 = np.sum(abs(np.array(z[1].cpu()) - np.array(outputs[1].cpu())))
            dif2 = np.sum(abs(np.array(z[2].cpu()) - np.array(outputs[2].cpu())))
            dif3 = np.sum(abs(np.array(z[3].cpu()) - np.array(outputs[3].cpu())))
            print('z recon error is {}:'.format((dif0 + dif1 + dif2 + dif3) / n / 8192))

            print('-----end-----' * 7)
        break