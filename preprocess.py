import os
import cv2
import numpy as np
from PIL import Image
from utils.metrics import calculate_psnr, calculate_ssim
import datasets
import yaml
import argparse
import sys
import torch
import datasets.all_light


config_path = 'all_light.yml'
test_set = 'lol_v1'
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
train_loader, val_loader = DATASET.get_loaders(parse_patches=False, validation=test_set)

train_root = os.path.join(getattr(config.path, test_set))
path_dict = getattr(datasets.all_light, f'get_{test_set}_root')(config)
train_input_root = path_dict['train_input_paths']
test_input_root = path_dict['test_input_paths']

train_results_root = os.path.join(os.path.dirname(train_input_root), 'clahe')
test_results_root = os.path.join(os.path.dirname(test_input_root), 'clahe')
os.makedirs(train_results_root, exist_ok=True)
os.makedirs(test_results_root, exist_ok=True)
print(train_input_root, train_results_root)
# exit()

data_path = '/home/cyf20/datasets/lol_dataset/our485/low'
img_name = '2.png'

# clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

def run_clahe(input_path, out_path, if_color=True):
    # img = cv2.resize(img, None, fx=0.5, fy=0.5)
    
    # 限制对比度的自适应阈值均衡化
    # if if_color:
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    image = cv2.imread(input_path)
    # cv2.imwrite(out_path.replace('.png', '1.png'), image, [cv2.IMWRITE_JPEG_QUALITY, 100])
    # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    b,g,r = cv2.split(image)
    # import pdb; pdb.set_trace()
    b = clahe.apply(b)
    g = clahe.apply(g)
    r = clahe.apply(r)
    image = cv2.merge([b,g,r])

    cv2.imwrite(out_path, image, [cv2.IMWRITE_JPEG_QUALITY, 100])

    
        # img_yuv = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
        # img_yuv[:,:,0] = cv2.equalizeHist(img_yuv[:,:,0])
        # equa = cv2.cvtColor(img_yuv,cv2.COLOR_YUV2BGR)

        # img_yuv2 = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
        # img_yuv2[:,:,0] =  clahe.apply(img_yuv[:,:,0])
        # dst = cv2.cvtColor(img_yuv2, cv2.COLOR_YUV2BGR)

    # else:
    #     img = cv2.imread(input_path)

    #     dst = clahe.apply(img)  # 限制对比度自适应直方图均衡化(CLAHE)
    #     # 使用全局直方图均衡化
    #     equa = cv2.equalizeHist(img) #全局直方图均衡化(HE)
    #     # 分别b原图，CLAHE，HE    灰度图处理
    #     cv2.imwrite('./493_clahe.png',dst)
    #     cv2.imwrite('./493_equa.png',equa)


import retinex
import json

# def msr(data_path,save_path):
#     img_list = os.listdir(data_path)
#     if len(img_list) == 0:
#         print ('Data directory is empty.')
#         exit()

#     with open('config.json', 'r') as f:
#         config = json.load(f)

#     for img_name in img_list:
#         img_name2 = img_name.split('.')[0]
#         if img_name == '.gitkeep':
#             continue
        
#         img = cv2.imread(os.path.join(data_path, img_name))

#         img_msrcr = retinex.MSRCR(
#             img,
#             config['sigma_list'],
#             config['G'],
#             config['b'],
#             config['alpha'],
#             config['beta'],
#             config['low_clip'],
#             config['high_clip']
#         )
    
#         img_amsrcr = retinex.automatedMSRCR(
#             img,
#             config['sigma_list']
#         )

#         img_msrcp = retinex.MSRCP(
#             img,
#             config['sigma_list'],
#             config['low_clip'],
#             config['high_clip']        
#         )    

#         shape = img.shape
#     #results_lol
        
#         cv2.imwrite(f'{save_path}/msrcr/{img_name2}.png',img_msrcr)



for loader, root, out_root in zip([train_loader, val_loader], [train_input_root, test_input_root], 
                                  [train_results_root, test_results_root]):
    for i, (x, y) in enumerate(loader):
        if isinstance(y, list):
            y = y[0]
        results_root = './'
        # results_path = os.path.join(results_root, f'{y}.png')
        input_path = os.path.join(root, f'{y}.png')

        results_path = os.path.join(out_root, f'{y}.png')
        # print(input_path, '?')
        run_clahe(input_path, results_path, True)
        # run_gauss(input_path, './gauss.png')
        # exit()

        

