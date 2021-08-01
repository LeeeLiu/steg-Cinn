#!/usr/bin/env python
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

# 这里就可以做映射了啊😀
# def sample_z(N, outputs):
#     sampled_z = []
#     for o in outputs:
#         shape = list(o.shape)
#         shape[0] = N
#         sampled_z.append(torch.randn(shape).cuda())
#     return sampled_z

# def cal_avg_var(z):
#     avg0 = np.average(np.array(z[0].cpu()))
#     avg1 = np.average(np.array(z[1].cpu()))
#     avg2 = np.average(np.array(z[2].cpu()))
#     avg3 = np.average(np.array(z[3].cpu()))
#     avg = (avg0 + avg1 + avg2 + avg3) / 4
#     var = np.var(np.array(z[0].cpu()))
#     return avg, var


# 嵌入

def sample_z(N, outputs, args):
    if args.mapping=='True':
        print('i am mapping, k={}'.format(args.size_k))
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

def reconZ_relate2which_jpg_is_small(z, outputs, batch_idx, n, args):
    stego2copy = './output/stego_png/k={}/batch={}/{}.png'

    stego_small_reconZ_dir = './output/stego_png/k={}_small_reconZ/batch={}'.format(args.size_k, batch_idx)
    if not os.path.exists(stego_small_reconZ_dir):
        os.makedirs(stego_small_reconZ_dir)

    copy_dir = './output/stego_png/k={}_small_reconZ/batch={}'

    log_file = './output/stego_png/k={}_small_reconZ/log.txt'.format(args.size_k)
    with open(log_file, 'a') as f:
        for i in range(n):
            dif = 0
            for k in range(4):
                dif += np.sum(abs(np.array(z[k][i].cpu()) - np.array(outputs[k][i].cpu())))
            dif = dif / 8192  # 64*64*2 = 8192
            f.write('batch={}, image_idx={}, reconZ_error={}\n'.format(batch_idx, i, dif))
            if(dif < 1.0e-05):
                cmd = 'cp '+stego2copy.format(args.size_k, batch_idx, i)+' '+copy_dir.format(args.size_k, batch_idx)
                os.system(cmd)

        f.close()


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

    # 数据X是Lab格式的
    batch_idx = 0
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

            print('反向 嵌入：采样的z[0] dtype is {}:'.format(z[0].dtype))
            # print('反向 嵌入：采样的z shape is {}:'.format(np.asarray(z).shape))
            print('1!!', x.shape)
            x_l, x_ab, cond, ab_pred = model.prepare_batch(x)
            x_ab_sampled = model.combined_model.module.reverse_sample(list(np.asarray(z)), cond)
            image = data.norm_lab_to_rgb(x_l, x_ab_sampled)
            print('反向 嵌入：生成的image shape is {}:'.format(image.shape))
            stego_dir = './output/stego_png/k={}/batch={}'.format(args.size_k, batch_idx)
            if not os.path.exists(stego_dir):
                os.makedirs(stego_dir)
            for i in range(image.shape[0]):
                imageio.imwrite(stego_dir+'/{}.png'.format(i), image[i].squeeze().transpose(1, 2, 0))
            # ims.extend(list(image))
            # viz.show_imgs(*ims)

            print('-------' * 10)
            # 正向 提取（方法：保存为图片，RGB整数存储）
            stego_list = sorted(glob.glob(join(stego_dir, '*.png')))
            stego_data = data.LabColorDataset(stego_list, None)
            stego_loader = DataLoader(stego_data, batch_size=n, shuffle=c.shuffle_val, num_workers=1,
                                     pin_memory=True, drop_last=False)
            for X_lab in stego_loader:
                print('2!!', X_lab.shape)
                L, ab, cnd, _ = model.prepare_batch(X_lab)

                print('condition[0]的误差是：', np.average(abs(np.array(cond[0].cpu()) - np.array(cnd[0].cpu()))))
                print('灰度L的误差是：', np.average(abs(np.array(x_l.cpu()) - np.array(L.cpu()))))
                print('色度x_ab的误差是：', np.average(abs(np.array(x_ab_sampled.cpu()) - np.array(ab.cpu()))))

                outputs = model.cinn(ab, cnd)
                print('正向 提取 由cinn得到的 z(outputs) shape is {}:'.format(np.asarray(outputs).shape))
                break

            # 一张3*256*256的图像是 8192= 2*64*64
            # 4张图像的Z 组成部分是 [4, 4096]，[4, 2048]，[4, 1536]，[4, 512]
            # 对于一张图：4096 + 2048 + 1536 + 512 - 2*64*64 = 0  good!!!
            # 前后Z（由4部分组成）的误差：
            # print('Z由4部分组成：',z[0].shape,z[1].shape,z[2].shape,z[3].shape)
            # 由统计得，Z是0均值1方差的。
            # print('\n', 'outputs的均值是：', cal_avg_var(outputs)[0])
            # print('outputs[0]的方差是：', cal_avg_var(outputs)[1], '\n')
            # np.savetxt('./z[0]-outputs[0].txt', np.array(z[0].cpu()) - np.array(outputs[0].cpu()))

            bitstream2 = extract(outputs, args)
            compare_bitstream(bitstream1, bitstream2, args)
            print('-----end-----' * 7)

        reconZ_relate2which_jpg_is_small(z, outputs, batch_idx, n, args)
        batch_idx += 1
        break