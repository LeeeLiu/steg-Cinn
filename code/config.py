#################
# Architecture: #
#################

# todo
real_dim = 128

img_dims_orig = (256, 256)
img_dims = (img_dims_orig[0] // 4, img_dims_orig[0] // 4)

# batch_size = 48
# device_ids = [0,1,2]

batch_size = 16
device_ids = [0]    # GPU ids. Set to [0] for single GPU

#############################
# Training hyperparameters: #
#############################
clamping = 1.5

load_inn_only = ''                  # only load the inn part of the architecture
load_feature_net = ''               # only load the cond. net part

# seed = 9287
import time
seed = int(time.time() * 256)

log10_lr = -4.0                     # Log learning rate
lr = 10**log10_lr
lr_feature_net = lr                 # lr of the cond. network

n_epochs = 120 * 4      # 第一次训练
n_its_per_epoch = 32 * 8            # In case the epochs should be cut short after n iterations

weight_decay = 1e-5
betas = (0.9, 0.999)                # concerning adam optimizer

init_scale = 0.030                  # initialization std. dev. of weights (0.03 is approx xavier)
pre_low_lr = 0                      # for the first n epochs, lower the lr by a factor of 20

#######################
# Dataset parameters: #
#######################

dataset = 'coco'                # also 'coco' is possible.
validation_images = './imagenet/validation_images.txt'
shuffle_train = True
shuffle_val = False
val_start = 0                       # use a slice [start:stop] of the entire val set
val_stop = 5120

end_to_end = True                   # Whether to leave the cond. net fixed
no_cond_net = False                 # Whether to use a cond. net at all
pretrain_epochs = 0                 # Train only the inn for n epochs before end-to-end

########################
# Display and logging: #
########################

sampling_temperature = 1.0          # latent std. dev. for preview images
loss_display_cutoff = 10            # cut off the loss so the plot isn't ruined
loss_names = ['L', 'lr']
preview_upscale = 256 // img_dims_orig[0]

img_folder = '../training'

silent = False
live_visualization = False
progress_bar = False

#######################
# Saving checkpoints: #
#######################

checkpoint_save_interval = 50       # 第一次训练
checkpoint_save_overwrite = False   # Whether to overwrite the old checkpoint with the new one
checkpoint_on_error = True          # Whether to make a checkpoint with suffix _ABORT if an error occurs

# 嵌入+提取时,
# 是否记录 RGB,Lab,Z 前后差距
record_reconZ = False
record_reconLab = False
record_reconRGB = False


# load_checkpoint = './checkpoints/11.26_cINN_3/ckpt.pt_checkpoint_0008'  # 【cINN 3】
# load_checkpoint = './checkpoints/11.25_cINN_2/ckpt.pt_checkpoint_0016'   # 【cINN 2】
# load_checkpoint = './checkpoints/11.23_uniformCkpt_with_genTIFF_augment/ckpt.pt_checkpoint_0030'    # 【cINN 1】
# load_checkpoint = './checkpoints/11.23_uniformCkpt_with_genTIFF_augment/ckpt.pt'
# load_checkpoint = './checkpoints/10.27_noise_uniform_with_augment/ckpt.pt'     # 【cINN 0】加噪-均匀

# load_checkpoint = './checkpoints/9.27_noise_Guassian_with_augment/ckpt_with_augmentation.pt'   # 加噪-高斯
# load_checkpoint = './pretrain_colorization_end2end_checkpoint_0160.pt'     # cINN原始的转换颜色方式


load_checkpoint = './checkpoints_backup/7.06_colourScience/ckpt.cINN_0'     # 【cINN 0】不加噪
# load_checkpoint = './checkpoints/12.30/ckpt.cINN_1'      #【cINN 1】
# load_checkpoint = './checkpoints/12.30/ckpt.cINN_2'      #【cINN 2】