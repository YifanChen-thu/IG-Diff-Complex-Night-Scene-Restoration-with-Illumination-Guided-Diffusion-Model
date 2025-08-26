import os
import sys
from os import listdir
from os.path import isfile
import torch
import numpy as np
import torchvision
import torch.utils.data
import PIL
import re
import random
import util
import glob

def divide_train_test(test_dir, subfolders_input, subfolders_gt, condition_root, if_extend=False):
    train_input_paths = []
    train_gt_paths = []
    test_input_paths = []
    test_gt_paths = []
    train_condition_paths = []
    test_condition_paths = []
    # subfolders_input = sorted(glob.glob(f'{input_path}/*/'))
    # subfolders_gt = sorted(glob.glob(f'{gt_path}/*/'))
    
    for subfolder_input, subfolder_gt in zip(subfolders_input, subfolders_gt):            
            subfolder_name = os.path.basename(subfolder_input)
            subfolder_name_gt = os.path.basename(subfolder_gt)
            assert subfolder_name == subfolder_name_gt
            # Test
            if (subfolder_name in test_dir):
                img_paths_input_test = util.glob_file_list(subfolder_input)
                img_paths_gt_test = util.glob_file_list(subfolder_gt)

                length_input = len(img_paths_input_test)
                if if_extend:
                    img_paths_gt_test.extend(img_paths_gt_test * (length_input-1))

                assert len(img_paths_input_test) == len(img_paths_gt_test)

                test_input_paths += img_paths_input_test
                test_gt_paths += img_paths_gt_test
            else:
                img_paths_input_train = util.glob_file_list(subfolder_input)
                img_paths_gt_train = util.glob_file_list(subfolder_gt)

                length_input = len(img_paths_input_train)
                if if_extend:
                    img_paths_gt_train.extend(img_paths_gt_train * (length_input - 1))
                
                assert len(img_paths_gt_train) == len(img_paths_input_train), \
                    f'{len(img_paths_gt_train)} != {len(img_paths_input_train)}'

                train_input_paths += img_paths_input_train
                train_gt_paths += img_paths_gt_train

    for path in train_input_paths:
        img_id_1 = re.split('/', path)[-2]
        img_id_2 = re.split('/', path)[-1]
        img_id_2_1 = re.split('.npy', img_id_2)[0]
        img_id = img_id_1 + '_' + img_id_2_1
        condition_path = os.path.join(condition_root, f'{img_id}.png')
        train_condition_paths.append(condition_path)

    for path in test_input_paths:
        img_id_1 = re.split('/', path)[-2]
        img_id_2 = re.split('/', path)[-1]
        img_id_2_1 = re.split('.npy', img_id_2)[0]
        img_id = img_id_1 + '_' + img_id_2_1
        condition_path = os.path.join(condition_root, f'{img_id}.png')
        test_condition_paths.append(condition_path)

    assert len(train_gt_paths) == len(train_condition_paths)
    assert len(test_gt_paths) == len(test_condition_paths)
    return train_input_paths, train_gt_paths, test_input_paths, test_gt_paths, train_condition_paths, test_condition_paths



def glob_path(path):
    return sorted(glob.glob(f'{path}/*.png')+glob.glob(f'{path}/*.jpg')+glob.glob(f'{path}/*.jpeg'))

def glob_path2(path):
    return sorted(glob.glob(f'{path}/*.png')+glob.glob(f'{path}/*.jpg')+glob.glob(f'{path}/*.jpeg'))
