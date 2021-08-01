#!/usr/bin/env python
import os
import time
import imageio
import numpy as np
import torch


# 版本说明
# 观察 原始、colourScience
# 生成图片 效果

def sample_z(N, outputs):
    sampled_z = []
    for o in outputs:
        shape = list(o.shape)
        shape[0] = N
        sample = torch.cuda.FloatTensor(shape[0], shape[1]).normal_()
        sampled_z.append(sample)
    return sampled_z

def if_path_not_exist_then_mkdir(dir):
    if not os.path.exists(dir):
        os.makedirs(dir)


def main(save_dir):
    batch_idx = 0
    for x, JPEGname, _ in data.test_loader:
        with torch.no_grad():
            n = x.shape[0]
            ims = []
            x = x[:n]
            print('-------' * 10)
            # 为了采样，先预测Z形状
            print('x.shape:', x.shape)
            xl, xab, xcnd, _ = model.prepare_batch(x)
            output_z = model.cinn(xab, xcnd)
            # 按照output_z的形状，生成N个样本
            # torch.manual_seed(c.seed)     # 随机种子旧位置
            z = sample_z(N=n, outputs=output_z)

            x_l, x_ab, cond, ab_pred = model.prepare_batch(x)
            x_ab_sampled = model.combined_model.module.reverse_sample(list(np.asarray(z)), cond)
            x_ab_sampled[:, 0].clamp_(-128, 128)
            x_ab_sampled[:, 1:].clamp_(-128, 128)

            rgb2save = data.norm_lab_to_rgb(x_l, x_ab_sampled)
            img2save = rgb2save

            print('保存之前，确认 dtype{}，shape{}'.format(img2save.dtype, img2save.shape))
            for i in range(img2save.shape[0]):
                ii = batch_idx * c.batch_size + i
                imageio.imwrite(save_dir + '/{}.png'.format(JPEGname[i][:-4]), img2save[i])

        print('------one batch end---- ' * 4)
        batch_idx += 1

        # if batch_idx == 10:
        #    break


if __name__ == '__main__':
    # 改1
    import config as c

    # import data_use_origin as data
    # import model_old as model
    # 改2
    import data
    import model

    # 改3
    # pretrain = './pretrain_colorization_end2end_checkpoint_0160.pt'
    # pretrain = '/dat01/liuting/proj/cINN/ckpt_backup/7.06_colourScience/ckpt.cINN_0'
    pretrain = '/dat01/liuting/proj/cINN/ckpt/Cinn_i_fromUniform/ckpt.cINN_5'

    model.load(pretrain)

    model.combined_model.eval()
    model.combined_model.module.inn.eval()
    model.combined_model.module.feature_network.eval()
    model.combined_model.module.fc_cond_network.eval()

    for i in range(20,30):
        # 改4
        save_dir = '../output/{}-Science_seed{}'.format(c.real_dim, i)
        if_path_not_exist_then_mkdir(save_dir)

        torch.manual_seed(i)  # 随机种子新位置
        main(save_dir)
