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
import cv2

from mapping_new_Hamming import bit2dec_new, dec2bit_new, compare_bitstream, compare

matplotlib.use('Agg')
import config as c
import data

if c.no_cond_net:
    import model_no_cond as model
else:
    import model

## 版本说明
    # 11-30_bitstream_embed_extract
    # 添加纠错码，只提取packet里面的data
        # 基于 11-25 版本
        # 载入哪一个checkpoint，在sh里传参    # todo

def sample_z(N, outputs, args):
    if args.mapping == 'True':
        print('i am mapping, k={}'.format(args.size_k))
        bitstream = ""  # 一维01字符串
        packet1 = ""  # 一维01字符串
        map_sampled_z = []  # 保持和outputs一样的维度（四部分组成）
        for o in outputs:
            shape = list(o.shape)
            shape[0] = N
            # 返回第一个是 data，第二个是 data+ECC # todo
            bits, data_ecc, z = bit2dec_new(shape, size_group=args.size_k, boundary=args.delta, gap=args.g / 2)
            bitstream += bits
            packet1 += data_ecc
            map_sampled_z.append(torch.tensor(z, dtype=torch.float32).cuda())
        return bitstream, map_sampled_z, packet1

    if args.mapping == 'False':
        sampled_z = []
        for o in outputs:
            shape = list(o.shape)
            shape[0] = N
            # 生成：标准正态分布，库函数
            # sampled_z.append(torch.cuda.FloatTensor(np.random.normal(0, 1, shape))) # 【1】
            sampled_z.append(torch.cuda.FloatTensor(shape[0], shape[1]).normal_())             # 【2】
            # sampled_z.append(torch.randn(shape).cuda())

        return "", sampled_z, ""

# 提取
def extract(multi_dim_latent, args):
    bitstream = ""  # 一维01字符串
    packet2 = ""
    if args.mapping:
        for noise in multi_dim_latent:
            noise = noise.squeeze().cpu().numpy()
            # 第一个没纠错的data+ECC，第二个是data。# todo
            DataEcc, bits = dec2bit_new(noise, args.size_k, args.g/2)
            bitstream += bits
            packet2 += DataEcc
    return bitstream, packet2


def if_path_not_exist_then_mkdir(dir):
    if not os.path.exists(dir):
        os.makedirs(dir)


def reconZ_relate2which_jpg_is_small(z, outputs, batch_idx, n, args, stego_dir):

    log_file = stego_dir+'/../log_reconZ.txt'
    with open(log_file, 'a') as f:
        jpg_idx_whose_reconZ_is_small = []
        for i in range(n):
            dif = 0
            for k in range(4):
                dif += np.sum(abs(np.array(z[k][i].cpu()) - np.array(outputs[k][i].cpu())))

                if c.record_reconZ:
                    # 09-21
                    DIR_check_z = stego_dir+ '/DIR_check_z'
                    DIR_check_z_hat = stego_dir + '/DIR_check_z_hat'
                    DIR_check_z_dif = stego_dir + '/DIR_check_z_dif'
                    if_path_not_exist_then_mkdir(DIR_check_z)
                    if_path_not_exist_then_mkdir(DIR_check_z_hat)
                    if_path_not_exist_then_mkdir(DIR_check_z_dif)

                    np.savetxt(DIR_check_z + '/image_idx={}_z[{}].txt'.format(i, k),
                               np.array(z[k][i].cpu()) )
                    np.savetxt(DIR_check_z_hat + '/image_idx={}_z_hat[{}].txt'.format(i, k),
                                np.array(outputs[k][i].cpu()))
                    np.savetxt(DIR_check_z_dif + '/image_idx={}_z_dif[{}].txt'.format(i, k),
                               np.array(z[k][i].cpu()) - np.array(outputs[k][i].cpu()))
                    # 09-21

            # 4张图像的Z: [4, 4096]，[4, 2048]，[4, 1536]，[4, 512]
            dif = dif / 8192      # 64*64*2 = 8192

            if (i // 10 == 0):
                ii = '0{}'.format(i)
            else:
                ii = '{}'.format(i)
            f.write('batch={}, image_idx={}：reconZ / dim={}\n'.format(batch_idx, ii, dif))

            # if(dif < 1 ):
            #     jpg_idx_whose_reconZ_is_small.append(i)
            #     print('I catch a {}({}) whose has small recon Z[0] !'.format(args.store_type, args.ColorSpace_type))
            #     img2copy = stego_dir+'/{}.{}'.format(ii, args.store_type)
            #     copy_dir = stego_dir+'/../small_reconZ/batch={}'.format(batch_idx)
            #     if not os.path.exists(copy_dir):
            #         os.makedirs(copy_dir)
            #     cmd = 'cp '+img2copy+' '+copy_dir
            #     os.system(cmd)

        f.close()
    # return jpg_idx_whose_reconZ_is_small


if __name__ == '__main__':
    ## 传参
    import argparse
    parser = argparse.ArgumentParser(description='cINN-based image steganography')
    parser.add_argument('--store_type', type=str, help='exr tiff png...', default='png')
    parser.add_argument('--ColorSpace_type', type=str, help='Lab RGB...', default='RGB')

    parser.add_argument('--mapping', type=str, help='map means embed msg')
    parser.add_argument('--size_k', type=int, help='k means size group(capacity)', default=1)
    parser.add_argument('--delta', type=float, help='delta means boundary', default=0.48)
    parser.add_argument('--g', type=float, help='g/2 means gap', default=0.01)

    parser.add_argument('--load_checkpoint', type=str, default=c.load_checkpoint)
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

    torch.manual_seed(c.seed)  # 随机种子新位置【外面】
    batch_idx = 0
    acc_list = []
    acc_list_corrected = []
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
            bitstream1, z, packet1= sample_z(N=n, outputs=output_z, args=args)

            print('反向 嵌入：采样的z[0] dtype is {}:'.format(z[0].dtype))
            # print('反向 嵌入：采样的z shape is {}:'.format(np.asarray(z).shape))
            x_l, x_ab, cond, ab_pred = model.prepare_batch(x)
            x_ab_sampled = model.combined_model.module.reverse_sample(list(np.asarray(z)), cond)
            x_ab_sampled[:, 0].clamp_(-128, 128)
            x_ab_sampled[:, 1:].clamp_(-128, 128)

            if args.mapping == 'True':
                stego_dir = './result/load_{}_synthesis_{}/delta={}/batch={}'\
                    .format(extension, args.store_type, args.delta, batch_idx)
            else:
                stego_dir = './result/load_{}_synthesis_{}/normal/batch={}' \
                    .format(extension, args.store_type, batch_idx)

            if_path_not_exist_then_mkdir(stego_dir)

            ## 把lab（float32）用 .exr 或者tiff存储
            ## 把RGB（float32）用 .exr 或者tiff存储
            ## 把RGB（png是uint8存储）用 .png存储
            if args.ColorSpace_type == 'Lab':
                import torch.nn.functional as F
                x_ab_sampled_256 = F.interpolate(x_ab_sampled, size=x_l.shape[2])  # , mode='bilinear', align_corners=False)
                lab2save = torch.cat([x_l, x_ab_sampled_256], dim=1)
                img2save = lab2save
            if args.ColorSpace_type == 'RGB':
                rgb2save = data.norm_lab_to_rgb(x_l, x_ab_sampled)
                img2save = rgb2save

                # 【取整】 09-21
                if args.store_type == 'png':
                    img2save = np.clip(img2save, 0, 255).round()

            print('一，保存{}之前，确认类型{}，形状{}'.format(args.ColorSpace_type, img2save.dtype, img2save.shape))
            for i in range(img2save.shape[0]):
                if( i//10 == 0):
                    ii = '0{}'.format(i)
                else:
                    ii = '{}'.format(i)

                # 【保存】
                imageio.imwrite(stego_dir + '/{}.{}'.format(ii, args.store_type), img2save[i])
                # 取整前RGB是`rgb2save`;
                # Clip+round取整后，imageio保存再读取，是`read_png`
                read_png = imageio.imread(stego_dir + '/{}.{}'.format(ii, args.store_type))

                if c.record_reconRGB:
                    # 09-21
                    DIR_check_rgb = stego_dir + '/DIR_check_rgb'
                    DIR_check_rgb_quant = stego_dir + '/DIR_check_rgb_hat'
                    DIR_check_rgb_dif = stego_dir + '/DIR_check_rgb_dif'
                    if_path_not_exist_then_mkdir(DIR_check_rgb)
                    if_path_not_exist_then_mkdir(DIR_check_rgb_quant)
                    if_path_not_exist_then_mkdir(DIR_check_rgb_dif)

                    for channel in range(3):
                        np.savetxt(DIR_check_rgb + '/image_idx={}_rgb[{}].txt'.format(i, channel),
                                   np.array(rgb2save[i].transpose(2, 0, 1)[channel]))
                        np.savetxt(DIR_check_rgb_quant + '/image_idx={}_rgb_quant[{}].txt'.format(i, channel),
                                   np.array(read_png.transpose(2, 0, 1)[channel]))
                        np.savetxt(DIR_check_rgb_dif + '/image_idx={}_rgb_dif[{}].txt'.format(i, channel),
                                   np.array(rgb2save[i].transpose(2, 0, 1)[channel]) - np.array(read_png.transpose(2, 0, 1)[channel]))
                    # 09-21

            # 正向 提取（保存为图片，RGB整数存储）
            print('-------' * 10)
            stego_list = sorted(glob.glob(join(stego_dir, '*.{}'.format(args.store_type))))
            stego_data = data.LabColorDataset(stego_list)
            stego_loader = DataLoader(stego_data, batch_size=n, shuffle=c.shuffle_val, num_workers=1,
                                      pin_memory=True, drop_last=False)
            for X_lab, _, _ in stego_loader:
                print('(2)从stego_loader读取的lab 形状', X_lab.shape)
                L, ab, cnd, _ = model.prepare_batch(X_lab)

                print('condition[0] 每维度误差：', np.average(abs(np.array(cond[0].cpu()) - np.array(cnd[0].cpu()))))
                print('灰度L 每维度误差：', np.average(abs(np.array(x_l.cpu()) - np.array(L.cpu()))))
                print('色度x_ab 每维度误差：', np.average(abs(np.array(x_ab_sampled.cpu()) - np.array(ab.cpu()))))

                if c.record_reconLab:
                    # 09-21
                    DIR_check_Lab = stego_dir + '/DIR_check_Lab'
                    DIR_check_Lab_hat = stego_dir + '/DIR_check_Lab_hat'
                    DIR_check_Lab_dif = stego_dir + '/DIR_check_Lab_dif'
                    if_path_not_exist_then_mkdir(DIR_check_Lab)
                    if_path_not_exist_then_mkdir(DIR_check_Lab_hat)
                    if_path_not_exist_then_mkdir(DIR_check_Lab_dif)
                    for i in range(x_ab_sampled.shape[0]):
                        np.savetxt(DIR_check_Lab + '/dif_L_{}.txt'.format(i), np.array(x_l[i].cpu() ).squeeze())
                        np.savetxt(DIR_check_Lab + '/dif_ab[0]_{}.txt'.format(i), np.array(x_ab_sampled[i][0].cpu() ))
                        np.savetxt(DIR_check_Lab + '/dif_ab[1]_{}.txt'.format(i), np.array(x_ab_sampled[i][1].cpu() ))

                        np.savetxt(DIR_check_Lab_hat + '/dif_L_{}.txt'.format(i), np.array( L[i].cpu()).squeeze())
                        np.savetxt(DIR_check_Lab_hat + '/dif_ab[0]_{}.txt'.format(i), np.array( ab[i][0].cpu()))
                        np.savetxt(DIR_check_Lab_hat + '/dif_ab[1]_{}.txt'.format(i), np.array( ab[i][1].cpu()))

                        np.savetxt(DIR_check_Lab_dif + '/dif_L_{}.txt'.format(i), np.array(x_l[i].cpu() - L[i].cpu()).squeeze())
                        np.savetxt(DIR_check_Lab_dif + '/dif_ab[0]_{}.txt'.format(i), np.array(x_ab_sampled[i][0].cpu() - ab[i][0].cpu()))
                        np.savetxt(DIR_check_Lab_dif + '/dif_ab[1]_{}.txt'.format(i), np.array(x_ab_sampled[i][1].cpu() - ab[i][1].cpu()))
                    # 09-21

                outputs = model.cinn(ab, cnd)
                print('正向 提取 由cinn得到的 z(outputs) shape is {}:'.format(np.asarray(outputs).shape))
                ###############################################

                reconZ_relate2which_jpg_is_small(z, outputs, batch_idx, n, args, stego_dir)
                if args.mapping == 'True':
                    bitstream2, packet2 = extract(outputs, args)
                    # 4张图像的Z: [4, 4096]，[4, 2048]，[4, 1536]，[4, 512]
                    # bitstream是一维度01字符串，长度[4, 8192]==[4*4096]+[4*2048]+[4*1536]+[4*512]
                    if(n*8192*(1/4) == len(bitstream2)):
                        print('提取比特流（packet里的data）长度正确')

                    # 四部分
                    print('extract bitstream from Z(all dim, one batch)')
                    len_bit = len(bitstream2)//2
                    acc_per_batch = compare_bitstream(bitstream1[:len_bit], bitstream2[:len_bit], args, stego_dir, batch_idx,  -1)
                    acc_list_corrected.append(acc_per_batch)

                    len_packet = len(packet1)//2
                    acc = compare(packet1[:len_packet], packet2[:len_packet])
                    acc_list.append(acc)

                    # print('extract bitstream from Z[0]')
                    # compare_bitstream(bitstream1[: n*4096], bitstream2[: n*4096],
                    #                   args, stego_dir, batch_idx,  0)
                    # print('extract bitstream from Z[1]')
                    # compare_bitstream(bitstream1[n*4096: n*(4096 + 2048)],
                    #                   bitstream2[n*4096: n*(4096 + 2048)],
                    #                   args, stego_dir, batch_idx,  1)
                    # print('extract bitstream from Z[2]')
                    # compare_bitstream(bitstream1[n * (4096+2048): n * (4096+2048 +1536)],
                    #                   bitstream2[n * (4096+2048): n * (4096+2048 +1536)],
                    #                   args, stego_dir, batch_idx,  2)
                    # print('extract bitstream from Z[3]')
                    # compare_bitstream(bitstream1[n * (4096+2048+1536): ],
                    #                   bitstream2[n * (4096+2048+1536): ],
                    #                   args, stego_dir, batch_idx,  3)

                    # 四部分
                else:
                    print('没有mapping')
                break

        print('------one batch end---- ' * 4)
        batch_idx += 1
        if batch_idx == 32:
            break

    f = open('./result/ACC/{}_delta={}_纠.txt'.format(extension, args.delta) , "w")
    f.write('{}'.format(  np.array(acc_list_corrected).mean() ))
    f.close()

    f = open('./result/ACC/{}_delta={}.txt'.format(extension, args.delta) , "w")
    f.write('{}'.format(  np.array(acc_list).mean() ))
    f.close()