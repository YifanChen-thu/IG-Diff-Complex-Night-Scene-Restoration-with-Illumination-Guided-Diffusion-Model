import os
import cv2
import numpy as np
from PIL import Image
from utils.metrics import calculate_psnr, calculate_ssim
import datasets
import yaml
import argparse
import torch

# Sample script to calculate PSNR and SSIM metrics from saved images in two directories
# using calculate_psnr and calculate_ssim functions from: https://github.com/JingyunLiang/SwinIR

# gt_path = '/PATH/TO/GROUND_TRUTH/'
# results_path = '/PATH/TO/MODEL_OUTPUTS/'


# test_set = 'sdsd_indoor'
# config_path = 'all_light_multi.yml'
# config_path = 'all_light.yml'
# test_set = 'sid'
# test_set = 'smid'
# results_root = 'test_results/sdsd_outdoor/sdsd_outdoor_5baseline_234w'
# results_root = 'test_results/sdsd_indoor/sdsd_indoor_5baseline_234w'
# results_root = 'test_results/sid/sid_baseline_141w_grid16'
# results_root = 'test_results/smid/smid_baseline_112w_grid16'
dataset_name = 'TCGA_LGG_GBM'

#patchDiff的结果
test_set = 't1t1ce'
config_path = 'all_light_medical_im_concat.yml'
results_root = 'test_results/TCGA_LGG_GBM/TCGA_LGG_GBM_2w'
log_file_name = f'test_results_metrics/{dataset_name}.txt'
other_format_results = 'PatchDiff'

#计算pallette的结果
##medical
test_set = 't1t1ce'
config_path = 'all_light_medical_im_concat.yml'
results_root = '../Palette-Image-to-Image-Diffusion-Models/experiments/test_tcga_lgg_gbm_240723_193155/results/test/0'
log_file_name = f'../Palette-Image-to-Image-Diffusion-Models/test_results_metrics/{dataset_name}.txt'
other_format_results = 'Palette'
# ##weather
# test_set = 'fog'
# config_path = 'all_light.yml'
# results_root = '../Palette-Image-to-Image-Diffusion-Models/experiments/test_tcga_lgg_gbm_240723_193155/results/test/0'
# log_file_name = f'../Palette-Image-to-Image-Diffusion-Models/test_results_metrics/{dataset_name}.txt'
# other_format_results = 'Palette'


# #计算i2imamba的结果
# dataset_name = 'LGG_T'  #改这里就不用改yml路径了
# test_set = 't1t1ce'  
# config_path = 'all_light_medical_im_concat.yml'
# results_root = f'../I2I-Mamba/results/{dataset_name}_t1_t1ce/test_latest/images'
# log_file_name = f'../I2I-Mamba/test_results_metrics/{dataset_name}.txt'
# other_format_results = 'I2I-Mamba'



# #TODO:考虑图像读取的通道数的问题
#计算mirnet的结果
# dataset_name = 'fog'
dataset_name = 'raindrop'
##weather
test_set = dataset_name
config_path = 'all_light_condition.yml'
results_root = f'../MIRNet-alllight/results/{dataset_name}/epoch4'
log_file_name = f'../MIRNet-alllight/test_results_metrics/{dataset_name}.txt'
other_format_results = 'MIRNet'

# CUDA_VISIBLE_DEVICES=5 python calculate_psnr_ssim_all_240723.py

verbose = False

def dict2namespace(config):
    namespace = argparse.Namespace()
    for key, value in config.items():
        if isinstance(value, dict):
            new_value = dict2namespace(value)
        else:
            new_value = value
        setattr(namespace, key, new_value)
    return namespace

with open(os.path.join("configs", config_path), "r") as f:
    config = yaml.safe_load(f)
config = dict2namespace(config)
config.data.data_name = test_set
# dn = config.path.t1t1ce.split('/')[-2]
# # import pdb;pdb.set_trace()
# sp_pre = config.path.t1t1ce.split(f'{dn}')[0]
# sp_last = config.path.t1t1ce.split(f'{dn}')[-1]
# config.path.t1t1ce = sp_pre + dataset_name + sp_last
# print(f'config.path.t1t1ce:{config.path.t1t1ce}')

print("=> using dataset '{}'".format(config.data.dataset))
DATASET = datasets.__dict__[config.data.dataset](config)
_, val_loader = DATASET.get_loaders(parse_patches=False, validation=test_set)


cumulative_psnr, cumulative_ssim, cnt = 0, 0, 0
assert len(val_loader) > 0

for i, (x, y) in enumerate(val_loader):
    if isinstance(y, list):
        y = y[0]
    
    # print()
    # folder =y.split('_')[0]
    # # print(folder)
    # y = folder + y.split(folder)[2]
    print(y)
    if other_format_results in ['PatchDiff', 'MIRNet']:
        results_path = os.path.join(results_root, f'{y}_output.png')
    elif other_format_results == 'Palette':
        results_path = os.path.join(results_root, f'Out_{y}.png')
    elif other_format_results == 'I2I-Mamba':
        results_path = os.path.join(results_root, f'{y}_fake_B.png')
    # results_path = os.path.join(results_root, f'{y}.png')
    
    gt = x[0, 3:]
    gt = gt.permute(1, 2, 0)

    W, H, _ = gt.shape

    gt = (gt * 255).clamp(0, 255).to(torch.uint8).detach().cpu().numpy()
    gt = Image.fromarray(gt)
    

    res = Image.open(results_path).convert('RGB').resize((H, W))
    # res = cv2.resize(res, (H, W), interpolation = cv2.INTER_CUBIC)

    gt = cv2.cvtColor(np.asarray(gt), cv2.COLOR_RGB2BGR)
    res = cv2.cvtColor(np.asarray(res), cv2.COLOR_RGB2BGR)
    # print(gt.size, res.size)

    cur_psnr = calculate_psnr(res, gt, test_y_channel=True)
    cur_ssim = calculate_ssim(res, gt, test_y_channel=True)
    if verbose:
        print(results_path)
        print('PSNR is %.4f and SSIM is %.4f' % (cur_psnr, cur_ssim))
    if cur_psnr != float('inf'):
        cumulative_psnr += cur_psnr
        cumulative_ssim += cur_ssim
        cnt += 1
        
if cnt > 0:
    val_psnr = cumulative_psnr / cnt
    val_ssim = cumulative_ssim / cnt
    print('Testing set, PSNR is %.4f and SSIM is %.4f' % (val_psnr, val_ssim))
    print(results_root)

os.makedirs(os.path.dirname(log_file_name), exist_ok=True)
with open(log_file_name, 'a') as f:
    f.write(f"dataset_input: {results_root} \n")
    f.write("Overall: PSNR {:4f} SSIM {:4f} \n".format(val_psnr, val_ssim))



