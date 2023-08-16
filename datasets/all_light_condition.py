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
import datasets.util as util
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
    return sorted(glob.glob(f'{path}/*.png')+glob.glob(f'{path}/*.jpg'))

def glob_path2(path):
    return sorted(glob.glob(f'{path}/*.jpg')+glob.glob(f'{path}/*.png'))

def check_path(train_input_paths, train_gt_paths):
    for (input, gt) in zip(train_input_paths, train_gt_paths):
        input = os.path.basename(input)
        gt = os.path.basename(gt)
        assert input == gt, f'[ERROR] {input} != {gt}'


def get_images_from_folders(subfolders_input, subfolders_gt, if_extend=False):
    input_paths=[]
    gt_paths=[]
    for subfolder_input, subfolder_gt in zip(subfolders_input, subfolders_gt):            
        subfolder_name = os.path.basename(subfolder_input)
        subfolder_name_gt = os.path.basename(subfolder_gt)
        assert subfolder_name == subfolder_name_gt
        
        img_paths_input = util.glob_file_list(subfolder_input)
        img_paths_gt = util.glob_file_list(subfolder_gt)

        length_input = len(img_paths_input)
        if if_extend:
            img_paths_gt.extend(img_paths_gt * (length_input-1))

        assert len(img_paths_input) == len(img_paths_gt)

        input_paths += img_paths_input
        gt_paths += img_paths_gt
        
    return input_paths, gt_paths

# 获取数据集的train_input_paths,train_gt_paths,test_input_paths,test_gt_paths
def lol_v1(config):
    # import pdb;pdb.set_trace()
    lol_v1_path = config.path.lol_v1

    train_input_paths = os.path.join(lol_v1_path,'our485','low')
    train_gt_paths = os.path.join(lol_v1_path,'our485','high')
    test_input_paths = os.path.join(lol_v1_path,'eval15','low')
    test_gt_paths = os.path.join(lol_v1_path,'eval15','high')
    train_condition_paths = os.path.join(lol_v1_path,'our485','sci_difficult_illu')
    test_condition_paths = os.path.join(lol_v1_path,'eval15','sci_difficult_illu')
    
    train_input_paths = glob_path(train_input_paths)
    train_gt_paths = glob_path(train_gt_paths)
    test_input_paths = glob_path(test_input_paths)
    test_gt_paths = glob_path(test_gt_paths)
    train_condition_paths = glob_path(train_condition_paths)
    test_condition_paths = glob_path(test_condition_paths)

    check_path(train_input_paths, train_gt_paths)
    check_path(test_input_paths, test_gt_paths)
    check_path(test_input_paths, test_condition_paths)

    return {'train_input_paths':train_input_paths,
        'train_gt_paths':train_gt_paths,
        'test_input_paths':test_input_paths,
        'test_gt_paths':test_gt_paths,
        'train_condition_paths': train_condition_paths,
        'test_condition_paths': test_condition_paths,
        }


def get_lol_v1_root(config):
    # import pdb;pdb.set_trace()
    lol_v1_path = config.path.lol_v1

    train_input_paths = os.path.join(lol_v1_path,'our485','low')
    train_gt_paths = os.path.join(lol_v1_path,'our485','high')
    test_input_paths = os.path.join(lol_v1_path,'eval15','low')
    test_gt_paths = os.path.join(lol_v1_path,'eval15','high')
    return {'train_input_paths':train_input_paths,
        'train_gt_paths':train_gt_paths,
        'test_input_paths':test_input_paths,
        'test_gt_paths':test_gt_paths} 


def lol_v2_real(config):
    lol_v2_real_path = config.path.lol_v2_real
    train_input_paths = os.path.join(lol_v2_real_path,'Train','Low')
    train_gt_paths = os.path.join(lol_v2_real_path,'Train','Normal')
    test_input_paths = os.path.join(lol_v2_real_path,'Test','Low')
    test_gt_paths = os.path.join(lol_v2_real_path,'Test','Normal')
    train_condition_paths = os.path.join(lol_v2_real_path, 'Train', 'sci_difficult_illu')
    test_condition_paths = os.path.join(lol_v2_real_path, 'Test', 'sci_difficult_illu')

    train_input_paths = glob_path(train_input_paths)
    train_gt_paths = glob_path(train_gt_paths)
    test_input_paths = glob_path(test_input_paths)
    test_gt_paths = glob_path(test_gt_paths)
    train_condition_paths = glob_path(train_condition_paths)
    test_condition_paths = glob_path(test_condition_paths)

    # check_path(train_input_paths, train_gt_paths)
    # check_path(test_input_paths, test_gt_paths)

    return {'train_input_paths':train_input_paths,
        'train_gt_paths':train_gt_paths,
        'test_input_paths':test_input_paths,
        'test_gt_paths':test_gt_paths,
        'train_condition_paths': train_condition_paths,
        'test_condition_paths': test_condition_paths,
        }


def lol_v2_syn(config):
    lol_v2_syn_path = config.path.lol_v2_syn
    train_input_paths = os.path.join(lol_v2_syn_path,'Train','Low')
    train_gt_paths = os.path.join(lol_v2_syn_path,'Train','Normal')
    test_input_paths = os.path.join(lol_v2_syn_path,'Test','Low')
    test_gt_paths = os.path.join(lol_v2_syn_path,'Test','Normal')
    train_condition_paths = os.path.join(lol_v2_syn_path, 'Train', 'sci_difficult_illu')
    test_condition_paths = os.path.join(lol_v2_syn_path, 'Test', 'sci_difficult_illu')

    train_input_paths = glob_path(train_input_paths)
    train_gt_paths = glob_path(train_gt_paths)
    test_input_paths = glob_path(test_input_paths)
    test_gt_paths = glob_path(test_gt_paths)
    train_condition_paths = glob_path(train_condition_paths)
    test_condition_paths = glob_path(test_condition_paths)

    # check_path(train_input_paths, train_gt_paths)
    # check_path(test_input_paths, test_gt_paths)

    return {'train_input_paths':train_input_paths,
        'train_gt_paths':train_gt_paths,
        'test_input_paths':test_input_paths,
        'test_gt_paths':test_gt_paths,
        'train_condition_paths': train_condition_paths,
        'test_condition_paths': test_condition_paths,
        }


def sdsd_indoor(config):
    sdsd_indoor_path = config.path.sdsd_indoor
    test_dir = config.test_dir.sdsd_indoor
    test_dir = test_dir.split(',')

    subfolders_input = util.glob_file_list(os.path.join(sdsd_indoor_path, 'input'))
    subfolders_gt = util.glob_file_list(os.path.join(sdsd_indoor_path, 'GT'))
    condition_root = os.path.join(sdsd_indoor_path, 'sci_difficult_illu_correct')

    train_input_paths, train_gt_paths, test_input_paths, test_gt_paths, train_condition_paths, test_condition_paths = \
        divide_train_test(test_dir, subfolders_input, subfolders_gt, condition_root, if_extend=False)

    assert len(train_input_paths) == len(train_gt_paths) == len(train_condition_paths)
    assert len(test_input_paths) == len(test_gt_paths) == len(test_condition_paths)

    return {'train_input_paths':train_input_paths,
        'train_gt_paths':train_gt_paths,
        'test_input_paths':test_input_paths,
        'test_gt_paths':test_gt_paths,
        'train_condition_paths': train_condition_paths,
        'test_condition_paths': test_condition_paths,
        }


def sdsd_outdoor(config):
    sdsd_outdoor_path = config.path.sdsd_outdoor
    test_dir = config.test_dir.sdsd_outdoor
    test_dir = test_dir.split(',')

    subfolders_input = util.glob_file_list(os.path.join(sdsd_outdoor_path,'input'))
    subfolders_gt = util.glob_file_list(os.path.join(sdsd_outdoor_path,'GT'))
    condition_root = os.path.join(sdsd_outdoor_path, 'sci_difficult_illu_correct')

    train_input_paths, train_gt_paths, test_input_paths, test_gt_paths, train_condition_paths, test_condition_paths = \
        divide_train_test(test_dir, subfolders_input, subfolders_gt, condition_root, if_extend=False)

    assert len(train_input_paths) == len(train_gt_paths) == len(train_condition_paths)
    assert len(test_input_paths) == len(test_gt_paths) == len(test_condition_paths)

    return {'train_input_paths':train_input_paths,
        'train_gt_paths':train_gt_paths,
        'test_input_paths':test_input_paths,
        'test_gt_paths':test_gt_paths,
        'train_condition_paths': train_condition_paths,
        'test_condition_paths': test_condition_paths,
        }


def sid(config):
    sid_path = config.path.sid

    subfolders_input = util.glob_file_list(os.path.join(sid_path,'short_sid2'))
    subfolders_gt =  util.glob_file_list(os.path.join(sid_path,'long_sid2'))
    condition_root = os.path.join(sid_path, 'sci_difficult_illu_correct')

    test_dir = []

    #test_namelist
    for mm in range(len(subfolders_input)):
        name = os.path.basename(subfolders_input[mm])
        if '1' in name[0]:
            test_dir.append(name)
      
    train_input_paths,train_gt_paths,test_input_paths,test_gt_paths, train_condition_paths, test_condition_paths = \
        divide_train_test(test_dir, subfolders_input, subfolders_gt, condition_root, if_extend=True)

    return {'train_input_paths':train_input_paths,
        'train_gt_paths':train_gt_paths,
        'test_input_paths':test_input_paths,
        'test_gt_paths':test_gt_paths,
        'train_condition_paths': train_condition_paths,
        'test_condition_paths': test_condition_paths,
        }


def smid(config):
    smid_path = config.path.smid
    test_dir = []
    f = open(config.test_dir.smid)
    lines = f.readlines()
    for mm in range(len(lines)):
        this_line = lines[mm].strip()
        test_dir.append(this_line)

    subfolders_input = util.glob_file_list(os.path.join(smid_path,'SMID_LQ_np'))
    subfolders_gt = util.glob_file_list(os.path.join(smid_path,'SMID_Long_np'))
    condition_root = os.path.join(smid_path, 'sci_difficult_illu_correct')

    train_input_paths,train_gt_paths,test_input_paths,test_gt_paths, train_condition_paths, test_condition_paths = \
        divide_train_test(test_dir,subfolders_input,subfolders_gt, condition_root,if_extend=True)

    return {'train_input_paths':train_input_paths,
        'train_gt_paths':train_gt_paths,
        'test_input_paths':test_input_paths,
        'test_gt_paths':test_gt_paths,
        'train_condition_paths': train_condition_paths,
        'test_condition_paths': test_condition_paths}
        
def lol_blur(config):
    lol_blur_path = config.path.lol_blur
    train_input_paths = os.path.join(lol_blur_path,'train','low_blur')
    train_gt_paths = os.path.join(lol_blur_path,'train','high_sharp_scaled')
    test_input_paths = os.path.join(lol_blur_path,'test','low_blur')
    test_gt_paths = os.path.join(lol_blur_path,'test','high_sharp_scaled')
    train_condition_paths = os.path.join(lol_blur_path, 'train', 'sci_difficult_illu_correct')
    # print(train_condition_paths)
    test_condition_paths = os.path.join(lol_blur_path, 'test', 'sci_difficult_illu_correct')

    train_input_folders = util.glob_file_list(train_input_paths)  #文件夹列出来
    train_gt_folders = util.glob_file_list(train_gt_paths)

    train_input_paths,train_gt_paths = get_images_from_folders(train_input_folders, train_gt_folders, if_extend=False)


    test_input_folders = util.glob_file_list(test_input_paths)
    test_gt_folders = util.glob_file_list(test_gt_paths)


    test_input_paths,test_gt_paths = get_images_from_folders(test_input_folders, test_gt_folders, if_extend=False)


    
    
    
    train_condition_paths = glob_path(train_condition_paths)  #图片列出来
    test_condition_paths = glob_path(test_condition_paths)
    

    # check_path(train_input_paths, train_gt_paths)
    # check_path(test_input_paths, test_gt_paths)

    return {'train_input_paths':train_input_paths,
        'train_gt_paths':train_gt_paths,
        'test_input_paths':test_input_paths,
        'test_gt_paths':test_gt_paths,
        'train_condition_paths': train_condition_paths,
        'test_condition_paths': test_condition_paths,
        }

def lol_blur_noise(config):
    lol_blur_noise_path = config.path.lol_blur_noise
    train_input_paths = os.path.join(lol_blur_noise_path,'train','low_blur_noise')
    train_gt_paths = os.path.join(lol_blur_noise_path,'train','high_sharp_scaled')
    test_input_paths = os.path.join(lol_blur_noise_path,'test','low_blur_noise')
    test_gt_paths = os.path.join(lol_blur_noise_path,'test','high_sharp_scaled')
    train_condition_paths = os.path.join(lol_blur_noise_path, 'train', 'lol_blur_noise_sci_difficult_illu_correct')
    test_condition_paths = os.path.join(lol_blur_noise_path, 'test', 'lol_blur_noise_sci_difficult_illu_correct')

    train_input_folders = util.glob_file_list(train_input_paths)  #文件夹列出来
    train_gt_folders = util.glob_file_list(train_gt_paths)

    train_input_paths,train_gt_paths = get_images_from_folders(train_input_folders, train_gt_folders, if_extend=False)


    test_input_folders = util.glob_file_list(test_input_paths)
    test_gt_folders = util.glob_file_list(test_gt_paths)


    test_input_paths,test_gt_paths = get_images_from_folders(test_input_folders, test_gt_folders, if_extend=False)


    
    
    
    train_condition_paths = glob_path(train_condition_paths)  #图片列出来
    test_condition_paths = glob_path(test_condition_paths)
    

    # check_path(train_input_paths, train_gt_paths)
    # check_path(test_input_paths, test_gt_paths)

    return {'train_input_paths':train_input_paths,
        'train_gt_paths':train_gt_paths,
        'test_input_paths':test_input_paths,
        'test_gt_paths':test_gt_paths,
        'train_condition_paths': train_condition_paths,
        'test_condition_paths': test_condition_paths,
        }

def lol_deblur_in_LDR(config):
    lol_blur_path = config.path.lol_blur
    train_input_paths = os.path.join(lol_blur_path,'train','low_blur')
    train_gt_paths = os.path.join(lol_blur_path,'train','high_sharp_scaled')
    test_input_paths = os.path.join(lol_blur_path,'test','low_blur')
    test_gt_paths = os.path.join(lol_blur_path,'test','high_sharp_scaled')
    train_condition_paths = os.path.join(lol_blur_path, 'train', 'sci_difficult_illu_correct')
    # print(train_condition_paths)
    test_condition_paths = os.path.join(lol_blur_path, 'test', 'sci_difficult_illu_correct')

    train_input_folders = util.glob_file_list(train_input_paths)  #文件夹列出来
    train_gt_folders = util.glob_file_list(train_gt_paths)

    train_input_paths,train_gt_paths = get_images_from_folders(train_input_folders, train_gt_folders, if_extend=False)


    test_input_folders = util.glob_file_list(test_input_paths)
    test_gt_folders = util.glob_file_list(test_gt_paths)


    test_input_paths,test_gt_paths = get_images_from_folders(test_input_folders, test_gt_folders, if_extend=False)


    #只改test_input_paths test_condition_paths
    test_input_paths = glob_path(config.path.lol_deblur_in_LDR)
    test_condition_paths = os.path.join('../../data/lol_deblur_in_LDR', 'test', 'sci_difficult_illu_correct')

    
    train_condition_paths = glob_path(train_condition_paths)  #图片列出来
    test_condition_paths = glob_path(test_condition_paths)


    
    

    # check_path(train_input_paths, train_gt_paths)
    # check_path(test_input_paths, test_gt_paths)

    return {'train_input_paths':train_input_paths,
        'train_gt_paths':train_gt_paths,
        'test_input_paths':test_input_paths,
        'test_gt_paths':test_gt_paths,
        'train_condition_paths': train_condition_paths,
        'test_condition_paths': test_condition_paths,
        }
def lol_deblur_in_MIMO(config):
    lol_blur_path = config.path.lol_blur
    train_input_paths = os.path.join(lol_blur_path,'train','low_blur')
    train_gt_paths = os.path.join(lol_blur_path,'train','high_sharp_scaled')
    test_input_paths = os.path.join(lol_blur_path,'test','low_blur')
    test_gt_paths = os.path.join(lol_blur_path,'test','high_sharp_scaled')
    train_condition_paths = os.path.join(lol_blur_path, 'train', 'sci_difficult_illu_correct')
    # print(train_condition_paths)
    test_condition_paths = os.path.join(lol_blur_path, 'test', 'sci_difficult_illu_correct')

    train_input_folders = util.glob_file_list(train_input_paths)  #文件夹列出来
    train_gt_folders = util.glob_file_list(train_gt_paths)

    train_input_paths,train_gt_paths = get_images_from_folders(train_input_folders, train_gt_folders, if_extend=False)


    test_input_folders = util.glob_file_list(test_input_paths)
    test_gt_folders = util.glob_file_list(test_gt_paths)


    test_input_paths,test_gt_paths = get_images_from_folders(test_input_folders, test_gt_folders, if_extend=False)


    #只改test_input_paths test_condition_paths
    test_input_paths = glob_path(config.path.lol_deblur_in_MIMO)
    test_condition_paths = os.path.join('../../data/lol_deblur_in_MIMO', 'test', 'sci_difficult_illu_correct')

    
    train_condition_paths = glob_path(train_condition_paths)  #图片列出来
    test_condition_paths = glob_path(test_condition_paths)


    
    

    # check_path(train_input_paths, train_gt_paths)
    # check_path(test_input_paths, test_gt_paths)

    return {'train_input_paths':train_input_paths,
        'train_gt_paths':train_gt_paths,
        'test_input_paths':test_input_paths,
        'test_gt_paths':test_gt_paths,
        'train_condition_paths': train_condition_paths,
        'test_condition_paths': test_condition_paths,
        }

def fog(config):
    # import pdb;pdb.set_trace()
    fog_path = config.path.fog

    train_input_paths = os.path.join(fog_path,'train','input')
    train_gt_paths = os.path.join(fog_path,'train','gt')
    test_input_paths = os.path.join(fog_path,'test','input')
    test_gt_paths = os.path.join(fog_path,'test','gt')
    train_condition_paths = os.path.join(fog_path, 'train', 'sci_difficult_illu_correct')
    # print(train_condition_paths)
    test_condition_paths = os.path.join(fog_path, 'test', 'sci_difficult_illu_correct')

    train_input_paths = glob_path(train_input_paths)
    train_gt_paths = glob_path(train_gt_paths)
    test_input_paths = glob_path(test_input_paths)
    test_gt_paths = glob_path(test_gt_paths)
    train_condition_paths = glob_path(train_condition_paths)
    test_condition_paths = glob_path(test_condition_paths)

    # check_path(train_input_paths, train_gt_paths)
    # check_path(test_input_paths, test_gt_paths)

    return {'train_input_paths':train_input_paths,
        'train_gt_paths':train_gt_paths,
        'test_input_paths':test_input_paths,
        'test_gt_paths':test_gt_paths,
        'train_condition_paths': train_condition_paths,
        'test_condition_paths': test_condition_paths,
        }
    
def raindrop(config):
    # import pdb;pdb.set_trace()
    raindrop_path = config.path.raindrop

    train_input_paths = os.path.join(raindrop_path,'train','input')
    train_gt_paths = os.path.join(raindrop_path,'train','gt')
    test_input_paths = os.path.join(raindrop_path,'test','input')
    test_gt_paths = os.path.join(raindrop_path,'test','gt')
    train_condition_paths = os.path.join(raindrop_path, 'train', 'sci_difficult_illu_correct')
    test_condition_paths = os.path.join(raindrop_path, 'test', 'sci_difficult_illu_correct')

    train_input_paths = glob_path(train_input_paths)
    train_gt_paths = glob_path(train_gt_paths)
    test_input_paths = glob_path(test_input_paths)
    test_gt_paths = glob_path(test_gt_paths)
    train_condition_paths = glob_path(train_condition_paths)
    test_condition_paths = glob_path(test_condition_paths)

    # check_path(train_input_paths, train_gt_paths)
    # check_path(test_input_paths, test_gt_paths)

    return {'train_input_paths':train_input_paths,
        'train_gt_paths':train_gt_paths,
        'test_input_paths':test_input_paths,
        'test_gt_paths':test_gt_paths,
        'train_condition_paths': train_condition_paths,
        'test_condition_paths': test_condition_paths,
        }
def raindrop_snow(config):
    # import pdb;pdb.set_trace()
    raindrop_path = config.path.raindrop
    raindrop_snow_path = config.path.raindrop_snow

    train_input_paths = os.path.join(raindrop_snow_path,'train','input_smallsnow')
    train_gt_paths = os.path.join(raindrop_path,'train','gt')
    test_input_paths = os.path.join(raindrop_snow_path,'test','input_smallsnow')
    test_gt_paths = os.path.join(raindrop_path,'test','gt')
    train_condition_paths = os.path.join(raindrop_snow_path, 'train', 'snowS_sci_difficult_illu_correct')
    test_condition_paths = os.path.join(raindrop_snow_path, 'test', 'snowS_sci_difficult_illu_correct')

    train_input_paths = glob_path(train_input_paths)
    train_gt_paths = glob_path(train_gt_paths)
    test_input_paths = glob_path(test_input_paths)
    test_gt_paths = glob_path(test_gt_paths)
    train_condition_paths = glob_path(train_condition_paths)
    test_condition_paths = glob_path(test_condition_paths)

    # check_path(train_input_paths, train_gt_paths)
    # check_path(test_input_paths, test_gt_paths)

    return {'train_input_paths':train_input_paths,
        'train_gt_paths':train_gt_paths,
        'test_input_paths':test_input_paths,
        'test_gt_paths':test_gt_paths,
        'train_condition_paths': train_condition_paths,
        'test_condition_paths': test_condition_paths,
        }
def rain(config):
    # import pdb;pdb.set_trace()
    rain_path = config.path.rain

    train_input_paths = os.path.join(rain_path,'train','input')
    train_gt_paths = os.path.join(rain_path,'train','gt')
    test_input_paths = os.path.join(rain_path,'test','input')
    test_gt_paths = os.path.join(rain_path,'test','gt')
    train_condition_paths = os.path.join(rain_path, 'train', 'sci_difficult_illu_correct')
    
    test_condition_paths = os.path.join(rain_path, 'test', 'sci_difficult_illu_correct')

    train_input_paths = glob_path2(train_input_paths)
    train_gt_paths = glob_path2(train_gt_paths)
    test_input_paths = glob_path2(test_input_paths)
    test_gt_paths = glob_path2(test_gt_paths)
    train_condition_paths = glob_path2(train_condition_paths)
    # print(len(train_condition_paths))
    test_condition_paths = glob_path2(test_condition_paths)

    # check_path(train_input_paths, train_gt_paths)
    # check_path(test_input_paths, test_gt_paths)

    return {'train_input_paths':train_input_paths,
        'train_gt_paths':train_gt_paths,
        'test_input_paths':test_input_paths,
        'test_gt_paths':test_gt_paths,
        'train_condition_paths': train_condition_paths,
        'test_condition_paths': test_condition_paths,
        }
def snow11k(config):
    # import pdb;pdb.set_trace()
    snow11k_path = config.path.snow11k

    train_input_paths = os.path.join(snow11k_path,'train','input')
    train_gt_paths = os.path.join(snow11k_path,'train','gt')
    test_input_paths = os.path.join(snow11k_path,'test','input')
    test_gt_paths = os.path.join(snow11k_path,'test','gt')
    train_condition_paths = os.path.join(snow11k_path, 'train', 'sci_difficult_illu_correct')
    print(train_condition_paths)
    test_condition_paths = os.path.join(snow11k_path, 'test', 'sci_difficult_illu_correct')
    
    train_input_paths = glob_path2(train_input_paths)
    train_gt_paths = glob_path2(train_gt_paths)
    test_input_paths = glob_path2(test_input_paths)
    test_gt_paths = glob_path2(test_gt_paths)
    train_condition_paths = glob_path(train_condition_paths)
    print(len(train_condition_paths))
    print(train_condition_paths)
    test_condition_paths = glob_path(test_condition_paths)

    # check_path(train_input_paths, train_gt_paths)
    # check_path(test_input_paths, test_gt_paths)

    return {'train_input_paths':train_input_paths,
        'train_gt_paths':train_gt_paths,
        'test_input_paths':test_input_paths,
        'test_gt_paths':test_gt_paths,
        'train_condition_paths': train_condition_paths,
        'test_condition_paths': test_condition_paths,}
def snow(config):
    # import pdb;pdb.set_trace()
    snow_path = config.path.snow

    train_input_paths = os.path.join(snow_path,'train','input')
    train_gt_paths = os.path.join(snow_path,'train','gt')
    test_input_paths = os.path.join(snow_path,'test','input')
    test_gt_paths = os.path.join(snow_path,'test','gt')
    train_condition_paths = os.path.join(snow_path, 'train', 'sci_difficult_illu_correct')
    test_condition_paths = os.path.join(snow_path, 'test', 'sci_difficult_illu_correct')

    train_input_paths = glob_path(train_input_paths)
    train_gt_paths = glob_path(train_gt_paths)
    test_input_paths = glob_path(test_input_paths)
    test_gt_paths = glob_path(test_gt_paths)
    train_condition_paths = glob_path(train_condition_paths)
    test_condition_paths = glob_path(test_condition_paths)

    # check_path(train_input_paths, train_gt_paths)
    # check_path(test_input_paths, test_gt_paths)

    return {'train_input_paths':train_input_paths,
        'train_gt_paths':train_gt_paths,
        'test_input_paths':test_input_paths,
        'test_gt_paths':test_gt_paths,
        'train_condition_paths': train_condition_paths,
        'test_condition_paths': test_condition_paths,
        }
def haze(config):
    # import pdb;pdb.set_trace()
    haze_path = config.path.haze

    train_input_paths = os.path.join(haze_path,'train','input')
    train_gt_paths = os.path.join(haze_path,'train','gt')
    test_input_paths = os.path.join(haze_path,'test','input')
    test_gt_paths = os.path.join(haze_path,'test','gt')
    train_condition_paths = os.path.join(haze_path, 'train', 'sci_difficult_illu_correct')
    test_condition_paths = os.path.join(haze_path, 'test', 'sci_difficult_illu_correct')

    train_input_paths = glob_path(train_input_paths)
    # train_gt_paths = []
    # for input_p in train_input_paths:
        
    train_gt_paths = glob_path(train_gt_paths)
    test_input_paths = glob_path(test_input_paths)
    test_gt_paths = glob_path(test_gt_paths)
    train_condition_paths = glob_path(train_condition_paths)
    test_condition_paths = glob_path(test_condition_paths)

    # check_path(train_input_paths, train_gt_paths)
    # check_path(test_input_paths, test_gt_paths)

    return {'train_input_paths':train_input_paths,
        'train_gt_paths':train_gt_paths,
        'test_input_paths':test_input_paths,
        'test_gt_paths':test_gt_paths,
        'train_condition_paths': train_condition_paths,
        'test_condition_paths': test_condition_paths,
        }

def real_weather(config):
    # import pdb;pdb.set_trace()
    real_weather_path = config.path.real_weather

    # train_input_paths = os.path.join(real_weather_path,'train','input')
    # train_gt_paths = os.path.join(real_weather_path,'train','gt')
    test_input_paths = os.path.join(real_weather_path,'test','input')
    # test_gt_paths = os.path.join(real_weather_path,'test','gt')
    
    # train_input_paths = glob_path(train_input_paths)
    # train_gt_paths = glob_path(train_gt_paths)
    test_input_paths = glob_path(test_input_paths)
    # test_gt_paths = glob_path(test_gt_paths)

    test_condition_paths = os.path.join(real_weather_path, 'test', 'sci_difficult_illu_correct')
    test_condition_paths = glob_path(test_condition_paths)

    return {'train_input_paths':test_input_paths,
        'train_gt_paths':test_input_paths,
        'test_input_paths':test_input_paths,
        'test_gt_paths':test_input_paths,
        'train_condition_paths': test_condition_paths,
        'test_condition_paths': test_condition_paths,}

class AllLightCondition:
    def __init__(self, config):
        self.config = config
        self.transforms = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])

    def get_loaders(self, parse_patches=True, validation='snow'):
        # import pdb;pdb.set_trace()
        datasetnames = self.config.data.data_name.split(',')
        
        train_input_paths = []
        train_gt_paths = []
        test_input_paths = []
        test_gt_paths = []
        train_condition_paths = []
        test_condition_paths = []
        #选择要合并训练的数据集名字
        for datasetname in datasetnames:
            #根据名字找函数
            #getattr(sys.modules[__name__], func_name)
            print(__name__, datasetname)
            path_dict = getattr(sys.modules[__name__], datasetname)(self.config)
            train_input_paths += path_dict['train_input_paths']
            train_gt_paths += path_dict['train_gt_paths']
            test_input_paths += path_dict['test_input_paths']
            test_gt_paths += path_dict['test_gt_paths']
            train_condition_paths += path_dict['train_condition_paths']
            test_condition_paths += path_dict['test_condition_paths']

        if self.config.data.testing_block is not None:
            testing_block = self.config.data.testing_block.split('/')
            index = int(testing_block[0])
            block_number = int(testing_block[1])
            block_length = len(test_input_paths) // block_number
            l = block_length * (index-1)
            r = block_length + l
            if index == block_number:
                r = len(test_input_paths)
            # train_input_paths = train_input_paths[l:r]
            # train_gt_paths = train_gt_paths[l:r]
            test_input_paths = test_input_paths[l:r]
            test_gt_paths = test_gt_paths[l:r]
            # train_condition_paths = train_condition_paths[l:r]
            test_condition_paths = test_condition_paths[l:r]
            print(f'Testing Block: {self.config.data.testing_block}, l: {l}, r: {r}')

        train_dataset = AllLightDataset(input_paths=train_input_paths,
                                        gt_paths=train_gt_paths,
                                        condition_paths=train_condition_paths,
                                          n=self.config.training.patch_n,
                                          patch_size=self.config.data.image_size,
                                          transforms=self.transforms,
                                          parse_patches=parse_patches)
        val_dataset = AllLightDataset(input_paths=test_input_paths,
                                        gt_paths=test_gt_paths,
                                        condition_paths=test_condition_paths,
                                        n=self.config.training.patch_n,
                                        patch_size=self.config.data.image_size,
                                        transforms=self.transforms,
                                        parse_patches=False)

        if not parse_patches:
            self.config.training.batch_size = 1
            self.config.sampling.batch_size = 1
            shuffle_train = False
        else:
            shuffle_train = True

        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=self.config.training.batch_size,
                                                   shuffle=shuffle_train, num_workers=self.config.data.num_workers,
                                                   pin_memory=True)
        val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=1,
                                                 shuffle=False, num_workers=self.config.data.num_workers,
                                                 pin_memory=True)

        return train_loader, val_loader


class AllLightDataset(torch.utils.data.Dataset):
    def __init__(self, input_paths, gt_paths, condition_paths, patch_size, n, transforms, parse_patches=True):
        super().__init__()

        self.input_paths = input_paths
        self.gt_paths = gt_paths
        self.condition_paths = condition_paths

        self.patch_size = patch_size
        self.transforms = transforms
        self.n = n
        self.parse_patches = parse_patches

    @staticmethod
    def get_params(img, output_size, n):
        w, h = img.size
        th, tw = output_size
        if w == tw and h == th:
            return 0, 0, h, w

        i_list = [random.randint(0, h - th) for _ in range(n)]
        j_list = [random.randint(0, w - tw) for _ in range(n)]
        return i_list, j_list, th, tw

    @staticmethod
    def n_random_crops(img, x, y, h, w):
        crops = []
        for i in range(len(x)):
            new_crop = img.crop((y[i], x[i], y[i] + w, x[i] + h))
            crops.append(new_crop)
        return tuple(crops)

    def get_images(self, index):
        # print(condition_paths)
        # print(index)
        input_path = self.input_paths[index]
        gt_path = self.gt_paths[index]
        condition_path = self.condition_paths[index]

        if input_path.endswith('npy'):
            input_img = np.load(input_path)
            input_img = input_img[:, :, [2, 1, 0]]
            input_img = PIL.Image.fromarray(input_img).resize((960, 512))  # from SNR

            img_id_1 = re.split('/', input_path)[-2]
            img_id_2 = re.split('/', input_path)[-1]
            img_id_2_1 = re.split('.npy', img_id_2)[0]
            img_id = img_id_1 + '_' + img_id_2_1

            gt_img = np.load(gt_path)
            gt_img = gt_img[:, :, [2, 1, 0]]
            gt_img = PIL.Image.fromarray(gt_img).resize((960, 512))  # from SNR

            condition_img = PIL.Image.open(condition_path).convert('RGB').resize((960, 512))

        elif re.split('/',input_path)[-5]=='LOL-Blur':
            input_img = PIL.Image.open(input_path).convert('RGB').resize((700, 400)) #1120*640 1.75
            # img_id = re.split('/', input_path)[-1][:-4]

            gt_img = PIL.Image.open(gt_path).convert('RGB').resize((700, 400))
            condition_img = PIL.Image.open(condition_path).convert('RGB').resize((700, 400))


            #lol-blur
            img_id_1 = re.split('/', input_path)[-2]
            img_id_2 = re.split('/', input_path)[-1]
            img_id_2_1 = re.split('.png', img_id_2)[0]
            img_id = img_id_1 + '_' + img_id_2_1
            
        else:
            input_img = PIL.Image.open(input_path).convert('RGB')
            img_id = re.split('/', input_path)[-1][:-4]

            gt_img = PIL.Image.open(gt_path).convert('RGB')
            condition_img = PIL.Image.open(condition_path).convert('RGB')


            # #lol-blur
            # img_id_1 = re.split('/', input_path)[-2]
            # img_id_2 = re.split('/', input_path)[-1]
            # img_id_2_1 = re.split('.png', img_id_2)[0]
            # img_id = img_id_1 + '_' + img_id_2_1




        if self.parse_patches:
            i, j, h, w = self.get_params(input_img, (self.patch_size, self.patch_size), self.n)
            input_img = self.n_random_crops(input_img, i, j, h, w)
            gt_img = self.n_random_crops(gt_img, i, j, h, w)
            condition_img = self.n_random_crops(condition_img, i, j, h, w)

            outputs = [torch.cat([self.transforms(input_img[i]), self.transforms(gt_img[i]), self.transforms(condition_img[i])], dim=0)
                       for i in range(self.n)]
            return torch.stack(outputs, dim=0), img_id
        else:
            # Resizing images to multiples of 16 for whole-image restoration   inference
            wd_new, ht_new = input_img.size
            if ht_new > wd_new and ht_new > 1024:
                wd_new = int(np.ceil(wd_new * 1024 / ht_new))
                ht_new = 1024
            elif ht_new <= wd_new and wd_new > 1024:
                ht_new = int(np.ceil(ht_new * 1024 / wd_new))
                wd_new = 1024
            wd_new = int(16 * np.ceil(wd_new / 16.0))
            ht_new = int(16 * np.ceil(ht_new / 16.0))

            input_img = input_img.resize((wd_new, ht_new), PIL.Image.ANTIALIAS)
            gt_img = gt_img.resize((wd_new, ht_new), PIL.Image.ANTIALIAS)
            condition_img = condition_img.resize((wd_new, ht_new), PIL.Image.ANTIALIAS)

            return torch.cat([self.transforms(input_img), self.transforms(gt_img), self.transforms(condition_img)], dim=0), img_id

    def __getitem__(self, index):
        res = self.get_images(index)
        return res

    def __len__(self):
        return len(self.input_paths)
