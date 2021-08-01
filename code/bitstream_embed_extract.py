#!/usr/bin/env python
import os
import time

import imageio
import matplotlib
import numpy as np
import torch

matplotlib.use('Agg')
import config as c
import data

if c.no_cond_net:
    import model_no_cond as model
else:
    import model

## 版本说明
    # 01-23_bitstream_embed_extract
    # 合并 batch
        # 基于 01-15_bitstream_embed_extract

# todo
# from mapping_LT import bit2dec_LT, dec2bit_LT, compare_bitstream
from mapping_new import bit2dec_new, dec2bit_new, compare_bitstream

def sample_z(N, outputs, args):
    if args.mapping == 'True':
        print('i am mapping, k={}'.format(args.size_k))
        bitstream = ""  # 一维01字符串
        map_sampled_z = []  # 保持和outputs一样的维度（四部分组成）
        start1 = time.time()
        for o in outputs:
            shape = list(o.shape)
            shape[0] = N
            bits, z = bit2dec_new(shape, args.size_k, args.delta, args.g/2, args)
            # bits, z = bit2dec_LT(shape, args.size_k, args.a)
            bitstream += bits
            map_sampled_z.append(torch.tensor(z, dtype=torch.float32).cuda())

        duration = time.time() - start1
        print("嵌入耗时 k={}【每个batch】  {:.3f} 秒".format(args.size_k, duration))
        return bitstream, map_sampled_z

    if args.mapping == 'False':
        sampled_z = []
        # start1 = time.time()
        for o in outputs:
            shape = list(o.shape)
            shape[0] = N

            if args.normalFunc == 'old':
                sample = torch.randn(shape).cuda()   # 标准正态分布
            if args.normalFunc == 'new':
                sample = torch.cuda.FloatTensor(shape[0], shape[1]).normal_()
            sampled_z.append(sample)

        # duration = time.time() - start1
        # print("normal之{}【2，64，64】耗时  {:.3f} 秒".format(args.normalFunc, duration))
        return "", sampled_z
# 提取
def extract(multi_dim_latent, args):
    bitstream = ""  # 一维01字符串
    if args.mapping:
        start1 = time.time()

        for noise in multi_dim_latent:
            noise = noise.squeeze().cpu().numpy()
            bits = dec2bit_new(noise, args.size_k, args.g/2)
            # bits = dec2bit_LT(noise, args.size_k)
            bitstream += bits

        duration = time.time() - start1
        print("提取耗时 k={}【每个batch】  {:.3f} 秒".format(args.size_k, duration))

    return bitstream

def if_path_not_exist_then_mkdir(dir):
    if not os.path.exists(dir):
        os.makedirs(dir)

import matplotlib.pyplot as plt
def plot_hist(z, z_hat, batch_idx, stego_dir):
    data0 = []
    for noise in z:
        noise = noise.squeeze().cpu().numpy().flatten()
        data0.extend(noise)
    plt.hist(data0, bins=100, rwidth=0.9, density=True)
    save_dir = stego_dir+'/../'
    plt.savefig(save_dir+'batch_idx={}_Hist__Z.png'.format(batch_idx))
    plt.clf()
##
    data = []
    for noise in z_hat:
        noise = noise.squeeze().cpu().numpy().flatten()
        data.extend(noise)

    plt.hist(data, bins=100, rwidth=0.9, density=True)
    save_dir = stego_dir + '/../'
    plt.savefig(save_dir + 'batch_idx={}_Hist__Z-hat.png'.format(batch_idx))
    plt.clf()


if __name__ == '__main__':
    ## 传参
    import argparse
    parser = argparse.ArgumentParser(description='cINN-based image steganography')
    parser.add_argument('--store_type', type=str, help='exr tiff png...', default='png')
    parser.add_argument('--ColorSpace_type', type=str, help='Lab RGB...', default='RGB')

    parser.add_argument('--mapping', type=str, help='map means embed msg')
    parser.add_argument('--a', type=float)

    parser.add_argument('--size_k', type=int, help='k means size group(capacity)', default=1)
    parser.add_argument('--add_2', type=str, default='N')

    parser.add_argument('--delta', type=float, help='delta means boundary')
    parser.add_argument('--g', type=float, help='g/2 means gap')

    parser.add_argument('--load_checkpoint', type=str)
    parser.add_argument('--normalFunc', type=str, default='new')
    args = parser.parse_args()
    ## 传参

    model_name = args.load_checkpoint
    model.load(model_name)
    model.combined_model.eval()
    model.combined_model.module.inn.eval()

    # checkpoint后缀记录的有epoch
    _, fullname = os.path.split(args.load_checkpoint)
    (name, extension) = os.path.splitext(fullname)

    if not c.no_cond_net:
        model.combined_model.module.feature_network.eval()
        model.combined_model.module.fc_cond_network.eval()


    torch.manual_seed(c.seed)  # 随机种子新位置
    batch_idx = 0
    acc_list = []
    for x, _, _ in data.test_loader:    # 数据X是Lab格式
        with torch.no_grad():
            n = x.shape[0]
            ims = []
            x = x[:n]
            print('-------' * 10)
            # 反向 嵌入
            # 为了采样，先预测Z形状
            print('x.shape:', x.shape)
            xl, xab, xcnd, _ = model.prepare_batch(x)
            output_z = model.cinn(xab, xcnd)
            # 按照output_z的形状，生成N个样本

            # torch.manual_seed(c.seed)     # 随机种子旧位置

            ## 嵌入比特流
            bitstream1, z = sample_z(N=n, outputs=output_z, args=args)

            print('反向 嵌入：采样的z[0] dtype is {}:'.format(z[0].dtype))
            # print('反向 嵌入：采样的z shape is {}:'.format(np.asarray(z).shape))
            x_l, x_ab, cond, ab_pred = model.prepare_batch(x)
            x_ab_sampled = model.combined_model.module.reverse_sample(list(np.asarray(z)), cond)
            x_ab_sampled[:, 0].clamp_(-128, 128)
            x_ab_sampled[:, 1:].clamp_(-128, 128)

            if args.mapping == 'True':                          # todo
                stego_dir = './output/result4/stego/{}_load{}_delta={}_{}'\
                    .format(args.store_type, extension, args.delta, args.add_2)
            else:
                stego_dir = './output/result4/cover/load{}'.format(extension)
            if_path_not_exist_then_mkdir(stego_dir)

            if args.ColorSpace_type == 'RGB':
                rgb2save = data.norm_lab_to_rgb(x_l, x_ab_sampled)
                img2save = rgb2save
                # 【取整】 09-21
                if args.store_type == 'png':
                    img2save = np.clip(img2save, 0, 255).round()

            print('一，保存{}之前，确认 dtype{}，shape{}'.format(args.ColorSpace_type, img2save.dtype, img2save.shape))
            X_lab = torch.rand(size=(c.batch_size, 3, 256, 256)).cuda()
            for i in range(img2save.shape[0]):
                ii = batch_idx*c.batch_size + i
                # if( i//10 == 0 ):
                #     ii = '0{}'.format(id)
                # else:
                #     ii = '{}'.format(id)
                # 【图像保存】
                if args.store_type == 'png':
                    imageio.imwrite(stego_dir + '/{}.{}'.format(ii, args.store_type), img2save[i])
                    read_png = imageio.imread(stego_dir + '/{}.{}'.format(ii, args.store_type))
                    print(args.store_type, '取整保存再读取，误差是', np.sum(np.array(img2save[i]) - np.array(read_png)) )
                    X_lab[i] = torch.cuda.FloatTensor(data.rgb_img_to_Lab(read_png))
                if args.store_type == 'tiff':
                    X_lab[i] = torch.cuda.FloatTensor(data.rgb_img_to_Lab(img2save[i]))

            # 正向 提取
            print('X_lab.shape', X_lab.shape)
            if args.mapping == 'True':
                L, ab, cnd, _ = model.prepare_batch(X_lab)
                # print('condition[0] 每维度误差：', np.average(abs(np.array(cond[0].cpu()) - np.array(cnd[0].cpu()))))
                # print('灰度L 每维度误差：', np.average(abs(np.array(x_l.cpu()) - np.array(L.cpu()))))
                # print('色度x_ab 每维度误差：', np.average(abs(np.array(x_ab_sampled.cpu()) - np.array(ab.cpu()))))
                outputs = model.cinn(ab, cnd)
                print('正向 提取 由cinn得到的 z(outputs) shape is {}:'.format(np.asarray(outputs).shape))
                # 绘制 cover或stego对应 Z，Z-hat 直方图  # todo
                # plot_hist(z, outputs, batch_idx, stego_dir)

                bitstream2 = extract(outputs, args)
                # 4张图像的Z: [4, 4096]，[4, 2048]，[4, 1536]，[4, 512]
                # k=1,n*8192
                print('比特流前后长度分别是：{}, {}'.format(len(bitstream1), len(bitstream2)))
                # bitstream是一维度01字符串，长度==[4*4096]+[4*2048]+[4*1536]+[4*512]
                print('extract bitstream from Z(all dim, one batch)')
                acc_per_batch = compare_bitstream(bitstream1, bitstream2, args, extension, batch_idx)
                acc_list.append(acc_per_batch)
            else:
                print('没有mapping')

        print('------one batch end---- ' * 4)
        batch_idx += 1
        # if batch_idx == 32:
        #     break

    # todo
    if args.mapping == 'True':
        f = open('./output/result4/ACC/{}_load{}_delta={}_{}.txt'
                 .format(args.store_type, extension, args.delta, args.add_2), "w")
        f.write('{}'.format(np.array(acc_list).mean()))
        f.close()