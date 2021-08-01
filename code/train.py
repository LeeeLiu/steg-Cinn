#!/usr/bin/env python
import os
import sys
import imageio
import torch
import torch.nn
import torch.optim
from torch.nn.functional import avg_pool2d#, interpolate
from torch.autograd import Variable
import numpy as np
import tqdm
import config_train as c

import argparse
parser = argparse.ArgumentParser(description='training')
parser.add_argument('--cINN_pre', type=int)
args = parser.parse_args()

import data_train as data       # todo

import viz
if c.no_cond_net:
    import model_no_cond as model
else:
    import model


class dummy_loss(object):
    def item(self):
        return 1.

def sample_outputs(sigma, out_shape):
    return [sigma * torch.cuda.FloatTensor(torch.Size((4, o))).normal_() for o in out_shape]

tot_output_size = 2 * c.img_dims[0] * c.img_dims[1]


# train.py
    # 版本说明：
    # 载入 增强样本（TIFF取整 + Z）
    # 基于 cINN_0 继续训练 11-23
    # 基于 cINN_1 继续训练 11-25
    # 基于 cINN_2 继续训练 11-26

try:
    load_ckpt = './checkpoints/01.17/ckpt.cINN_{}'.format(args.cINN_pre)
    model.load(load_ckpt)
    # model.load(c.ckpt_version)  # todo
    train_loader = data.get_train_loader(args.cINN_pre)

    min_epoch_loss = sys.maxsize
    ckpt_overwrite = False
    for i_epoch in range(-c.pre_low_lr, c.n_epochs):
        loss_history = []
        if i_epoch < 0:
            for param_group in model.optim.param_groups:
                param_group['lr'] = c.lr * 2e-2
        if i_epoch == 0:
            for param_group in model.optim.param_groups:
                param_group['lr'] = c.lr

        if c.end_to_end and i_epoch <= c.pretrain_epochs:
            for param_group in model.feature_optim.param_groups:
                param_group['lr'] = 0
            if i_epoch == c.pretrain_epochs:
                for param_group in model.feature_optim.param_groups:
                    param_group['lr'] = 1e-4

        i_batch = 0
        if c.is_augmented_genTIFF:
            for (x, label_Z) in train_loader:
                i_batch += 1

                z_from_uint8_img, zz, jac = model.combined_model(x)
                NLL_1 = 0.5 * zz - jac

                # z_dif = (z_from_uint8_img - label_Z)     # 列表不能直接相减
                recon_z = sum(torch.sum((o1-o2)**2, dim=1) for o1,o2 in zip(z_from_uint8_img, label_Z))

                l = torch.mean(NLL_1) / tot_output_size +\
                    torch.mean(recon_z) / tot_output_size

                l.backward()

                print('\n')
                print('loss第一部分：NLL_1', torch.mean(NLL_1) / tot_output_size)
                print('loss第二部分：recon_z（平方）', torch.mean(recon_z) / tot_output_size)

                model.optim_step()
                loss_history.append([l.item(), 0.])

        else:
            for x in train_loader:
                i_batch += 1

                zz, jac = model.combined_model(x)
                neg_log_likeli = 0.5 * zz - jac
                l = torch.mean(neg_log_likeli) / tot_output_size
                l.backward()

                model.optim_step()
                loss_history.append([l.item(), 0.])

        epoch_losses = np.mean(np.array(loss_history), axis=0)

        # # 记录最小loss
        # if epoch_losses[0] < min_epoch_loss:
        #     ckpt_overwrite = True
        #     min_epoch_loss = epoch_losses[0]
        # else:
        #     ckpt_overwrite = False
        # # 记录最小loss

        epoch_losses[1] = np.log10(model.optim.param_groups[0]['lr'])
        for i in range(len(epoch_losses)):
            epoch_losses[i] = min(epoch_losses[i], c.loss_display_cutoff)

        print('i_epoch=', i_epoch)      # 暂时 注释掉 实时观察 生成图片质量 ‘output/images/eval’

        # with torch.no_grad():
        #     ims = []
        #     for x, _ in realDat.test_loader:
        #         x_l, x_ab, cond, ab_pred = model.prepare_batch(x[:4])
        #
        #         z = sample_outputs(c.sampling_temperature, model.output_dimensions)
        #         #  combined_model 反向,  没有 ab加噪 操作
        #         x_ab_sampled = model.combined_model.module.reverse_sample(z, cond)
        #         im = [one_img.transpose(2, 0, 1) for one_img in realDat.norm_lab_to_rgb(x_l, x_ab_sampled)]
        #
        #         for i in range(4):
        #             imageio.imwrite(c.img_folder+'/{}_{}_{}.png'.format(i_epoch, i_batch, i), im[i].transpose(1, 2, 0))
        #             # 如果没有dtype, jpg显示是白色的
        #             ims.extend(list(np.array(im, dtype=np.uint8)))
        #         break
        #     viz.show_imgs(*ims)

        if i_epoch >= c.pretrain_epochs * 2:
            model.weight_scheduler.step(epoch_losses[0])
            model.feature_scheduler.step(epoch_losses[0])

        viz.show_loss(epoch_losses)

        ckpt_dir, _ = os.path.split(c.filename)
        if not os.path.exists(ckpt_dir):
            os.makedirs(ckpt_dir)

        # # 保存最好的ckpt
        # if i_epoch > 0 and ckpt_overwrite:
        #     model.save(c.filename)
        # # 保存最好的ckpt

        if i_epoch > 0 and (i_epoch % c.checkpoint_save_interval) == 0:
            model.save(c.filename + '_checkpoint_%.4i' % (i_epoch * (1-c.checkpoint_save_overwrite)))

    # model.save(c.filename)      # todo
    save_ckpt = './checkpoints/01.17/ckpt.cINN_{}'.format(args.cINN_pre + 1)
    model.save(save_ckpt)


except:
    if c.checkpoint_on_error:
        print('checkpoint ABORT.')
        # model.save(c.filename + '_ABORT')
    raise
finally:
    viz.signal_stop()
