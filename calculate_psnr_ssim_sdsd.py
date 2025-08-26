import os
import cv2
import numpy as np
from PIL import Image
from utils.metrics import calculate_psnr, calculate_ssim
import datasets
import yaml
import argparse
import torch
import tqdm
# Sample script to calculate PSNR and SSIM metrics from saved images in two directories
# using calculate_psnr and calculate_ssim functions from: https://github.com/JingyunLiang/SwinIR

# gt_path = '/PATH/TO/GROUND_TRUTH/'
# results_path = '/PATH/TO/MODEL_OUTPUTS/'

test_set = 'lol_blur'
gt_root = '../../data/LOL-Blur/test/high_sharp_scaled/'
# results_root_list = [
#     'final_results/lol_blur/llol_blur_cond_222w'
# ]


# # test_set = 'sdsd_indoor'
# test_set = 'sdsd_outdoor'
# # test_set = 'sid'
config_path = 'all_light_condition.yml'
# # config_path = 'all_light_condition_multi.yml'
# # test_set = 'lol_v2_real'
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

print("=> using dataset '{}'".format(config.data.dataset))
DATASET = datasets.__dict__[config.data.dataset](config)
_, val_loader = DATASET.get_loaders(parse_patches=False, validation=test_set)


# results_root = 'final_results/lol_blur/lol_blur_cond_240w'   
# results_root = 'final_results/sdsd_indoor/sdsd_indoor_cond_145w'
results_root_list = [
'final_results/lol_blur/lol_blur_cond_240w'
# 'final_results/lol_blur/lol_blur_cond_300w',
# 'final_results/lol_blur/lol_blur_cond_144w',
# 'final_results/lol_blur/lol_blur_cond_372w',
# 'final_results/lol_blur/lol_blur_cond_222w',
# 'final_results/lol_blur/lol_blur_cond_101w',
# 'final_results/lol_blur/lol_blur_mini_cond_222w',            
]

# def name_mapper(gt_name, mode):
#     if mode == 'lol_v2_real':
#         img_name = gt_name.replace('normal', 'low') + '_output.png'
#     elif mode == 'lol_blur':
#         img_name = f'{gt_name}.png'
#     elif mode == 'snow':
#         img_name = f'{gt_name}.png'
#     elif mode == 'rain':
#         img_name = f'{gt_name}.png'
#     # elif mode == 'haze':
#     #     img_name = f'{gt_name}.png'
#     elif mode == 'raindrop':
#         img_name = gt_name.replace('clean', 'rain') + '_output.png'
#     elif mode == 'fog':
#         img_name = gt_name.replace('clean', 'hazy') + '_output.png'

#     else:
#         img_name = f'{gt_name}_output.png'
#     return img_name


cumulative_psnr, cumulative_ssim, cnt = 0, 0, 0
assert len(val_loader) > 0

for results_root in results_root_list:
    print(f'[LOG]: {results_root}')

    for i, (x, y) in tqdm.tqdm(enumerate(val_loader)):
        if isinstance(y, list):
            y = y[0]
        
        # print(results_root, y)
        # results_path = os.path.join(results_root, f'{y}.png')
        results_path = os.path.join(results_root, f'{y}_output.png')
        # results_path = os.path.join(results_root, f'{y}_output_output.png')
        
        # gt = x[0, 3:]
        gt = x[0, 3:6] #lol_blur_condi
        gt = gt.permute(1, 2, 0)

        # W, H, _ = gt.shape

        gt = (gt * 255).clamp(0, 255).to(torch.uint8).detach().cpu().numpy()
        gt = Image.fromarray(gt)
        

        res = Image.open(results_path).convert('RGB') #.resize((H, W))
        W, H = res.size
        # # print(W,H)
        # gt = gt.resize((W, H))  #res<-gt
        # W, H, _ = res.shape
        # res = cv2.resize(res, (H, W), interpolation = cv2.INTER_CUBIC)

        gt = cv2.cvtColor(np.asarray(gt), cv2.COLOR_RGB2BGR)
        res = cv2.cvtColor(np.asarray(res), cv2.COLOR_RGB2BGR)
        # print(gt.size, res.size)

        cur_psnr = calculate_psnr(res, gt, test_y_channel=True)
        cur_ssim = calculate_ssim(res, gt, test_y_channel=True)
        if verbose:
            print(results_path)
            print('PSNR is %.4f and SSIM is %.4f' % (cur_psnr, cur_ssim))
        cumulative_psnr += cur_psnr
        cumulative_ssim += cur_ssim
        cnt += 1
        
    print('Testing set, PSNR is %.4f and SSIM is %.4f' % (cumulative_psnr / cnt, cumulative_ssim / cnt))
    print(results_root)

