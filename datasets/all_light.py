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


def divide_train_test(test_dir, subfolders_input, subfolders_gt, if_extend=False):
    train_input_paths = []
    train_gt_paths = []
    test_input_paths = []
    test_gt_paths = []

    # subfolders_input = sorted(glob.glob(f'{input_path}/*/'))
    # subfolders_gt = sorted(glob.glob(f'{gt_path}/*/'))
    
    for subfolder_input, subfolder_gt in zip(subfolders_input, subfolders_gt):            
            subfolder_name = os.path.basename(subfolder_input)
            subfolder_name_gt = os.path.basename(subfolder_gt)
            assert subfolder_name == subfolder_name_gt
            #test
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

    return train_input_paths, train_gt_paths, test_input_paths, test_gt_paths


def glob_path(path):
    return sorted(glob.glob(f'{path}/*.png'))

def glob_path2(path):
    return sorted(glob.glob(f'{path}/*.jpg'))

def check_path(train_input_paths, train_gt_paths):
    for (input, gt) in zip(train_input_paths, train_gt_paths):
        input = os.path.basename(input)
        gt = os.path.basename(gt)
        assert input == gt, f'[ERROR] {input} != {gt}'


# 获取数据集的train_input_paths,train_gt_paths,test_input_paths,test_gt_paths
def lol_v1(config):
    # import pdb;pdb.set_trace()
    lol_v1_path = config.path.lol_v1

    train_input_paths = os.path.join(lol_v1_path,'our485','low')
    train_gt_paths = os.path.join(lol_v1_path,'our485','high')
    test_input_paths = os.path.join(lol_v1_path,'eval15','low')
    test_gt_paths = os.path.join(lol_v1_path,'eval15','high')
    
    train_input_paths = glob_path(train_input_paths)
    train_gt_paths = glob_path(train_gt_paths)
    test_input_paths = glob_path(test_input_paths)
    test_gt_paths = glob_path(test_gt_paths)

    check_path(train_input_paths, train_gt_paths)
    check_path(test_input_paths, test_gt_paths)

    return {'train_input_paths':train_input_paths,
        'train_gt_paths':train_gt_paths,
        'test_input_paths':test_input_paths,
        'test_gt_paths':test_gt_paths}


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

    train_input_paths = glob_path(train_input_paths)
    train_gt_paths = glob_path(train_gt_paths)
    test_input_paths = glob_path(test_input_paths)
    test_gt_paths = glob_path(test_gt_paths)

    # check_path(train_input_paths, train_gt_paths)
    # check_path(test_input_paths, test_gt_paths)

    return {'train_input_paths':train_input_paths,
        'train_gt_paths':train_gt_paths,
        'test_input_paths':test_input_paths,
        'test_gt_paths':test_gt_paths}


def lol_v2_syn(config):
    lol_v2_syn_path = config.path.lol_v2_syn
    train_input_paths = os.path.join(lol_v2_syn_path,'Train','Low')
    train_gt_paths = os.path.join(lol_v2_syn_path,'Train','Normal')
    test_input_paths = os.path.join(lol_v2_syn_path,'Test','Low')
    test_gt_paths = os.path.join(lol_v2_syn_path,'Test','Normal')

    train_input_paths = glob_path(train_input_paths)
    train_gt_paths = glob_path(train_gt_paths)
    test_input_paths = glob_path(test_input_paths)
    test_gt_paths = glob_path(test_gt_paths)

    # check_path(train_input_paths, train_gt_paths)
    # check_path(test_input_paths, test_gt_paths)

    return {'train_input_paths':train_input_paths,
        'train_gt_paths':train_gt_paths,
        'test_input_paths':test_input_paths,
        'test_gt_paths':test_gt_paths}


def sdsd_indoor(config):
    sdsd_indoor_path = config.path.sdsd_indoor
    test_dir = config.test_dir.sdsd_indoor
    test_dir = test_dir.split(',')

    subfolders_input = util.glob_file_list(os.path.join(sdsd_indoor_path, 'input'))
    subfolders_gt = util.glob_file_list(os.path.join(sdsd_indoor_path, 'GT'))

    train_input_paths,train_gt_paths,test_input_paths,test_gt_paths = \
        divide_train_test(test_dir, subfolders_input, subfolders_gt, if_extend=False)

    return {'train_input_paths':train_input_paths,
        'train_gt_paths':train_gt_paths,
        'test_input_paths':test_input_paths,
        'test_gt_paths':test_gt_paths}


def sdsd_outdoor(config):
    sdsd_outdoor_path = config.path.sdsd_outdoor
    test_dir = config.test_dir.sdsd_outdoor
    test_dir = test_dir.split(',')

    subfolders_input = util.glob_file_list(os.path.join(sdsd_outdoor_path,'input'))
    subfolders_gt = util.glob_file_list(os.path.join(sdsd_outdoor_path,'GT'))

    train_input_paths,train_gt_paths,test_input_paths,test_gt_paths = \
        divide_train_test(test_dir, subfolders_input, subfolders_gt, if_extend=False)

    return {'train_input_paths':train_input_paths,
        'train_gt_paths':train_gt_paths,
        'test_input_paths':test_input_paths,
        'test_gt_paths':test_gt_paths}


def sid(config):
    sid_path = config.path.sid

    subfolders_input = util.glob_file_list(os.path.join(sid_path,'short_sid2'))
    subfolders_gt =  util.glob_file_list(os.path.join(sid_path,'long_sid2'))

    test_dir = []

    #test_namelist
    for mm in range(len(subfolders_input)):
        name = os.path.basename(subfolders_input[mm])
        if '1' in name[0]:
            test_dir.append(name)
      
    train_input_paths,train_gt_paths,test_input_paths,test_gt_paths = \
        divide_train_test(test_dir,subfolders_input,subfolders_gt,if_extend=True)

    return {'train_input_paths':train_input_paths,
        'train_gt_paths':train_gt_paths,
        'test_input_paths':test_input_paths,
        'test_gt_paths':test_gt_paths}


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

    train_input_paths,train_gt_paths,test_input_paths,test_gt_paths = \
        divide_train_test(test_dir,subfolders_input,subfolders_gt,if_extend=True)

    return {'train_input_paths':train_input_paths,
        'train_gt_paths':train_gt_paths,
        'test_input_paths':test_input_paths,
        'test_gt_paths':test_gt_paths}
def fog(config):
    # import pdb;pdb.set_trace()
    fog_path = config.path.fog

    train_input_paths = os.path.join(fog_path,'train','input')
    train_gt_paths = os.path.join(fog_path,'train','gt')
    test_input_paths = os.path.join(fog_path,'test','input')
    test_gt_paths = os.path.join(fog_path,'test','gt')
    
    train_input_paths = glob_path(train_input_paths)
    train_gt_paths = glob_path(train_gt_paths)
    test_input_paths = glob_path(test_input_paths)
    test_gt_paths = glob_path(test_gt_paths)

    # check_path(train_input_paths, train_gt_paths)
    # check_path(test_input_paths, test_gt_paths)

    return {'train_input_paths':train_input_paths,
        'train_gt_paths':train_gt_paths,
        'test_input_paths':test_input_paths,
        'test_gt_paths':test_gt_paths}
def raindrop(config):
    # import pdb;pdb.set_trace()
    raindrop_path = config.path.raindrop

    train_input_paths = os.path.join(raindrop_path,'train','input')
    train_gt_paths = os.path.join(raindrop_path,'train','gt')
    test_input_paths = os.path.join(raindrop_path,'test','input')
    test_gt_paths = os.path.join(raindrop_path,'test','gt')
    
    train_input_paths = glob_path(train_input_paths)
    train_gt_paths = glob_path(train_gt_paths)
    test_input_paths = glob_path(test_input_paths)
    test_gt_paths = glob_path(test_gt_paths)

    # check_path(train_input_paths, train_gt_paths)
    # check_path(test_input_paths, test_gt_paths)

    return {'train_input_paths':train_input_paths,
        'train_gt_paths':train_gt_paths,
        'test_input_paths':test_input_paths,
        'test_gt_paths':test_gt_paths}
def rain(config):
    # import pdb;pdb.set_trace()
    rain_path = config.path.rain

    train_input_paths = os.path.join(rain_path,'train','input')
    train_gt_paths = os.path.join(rain_path,'train','gt')
    test_input_paths = os.path.join(rain_path,'test','input')
    test_gt_paths = os.path.join(rain_path,'test','gt')
    
    train_input_paths = glob_path2(train_input_paths)
    train_gt_paths = glob_path2(train_gt_paths)
    test_input_paths = glob_path2(test_input_paths)
    test_gt_paths = glob_path2(test_gt_paths)

    # check_path(train_input_paths, train_gt_paths)
    # check_path(test_input_paths, test_gt_paths)

    return {'train_input_paths':train_input_paths,
        'train_gt_paths':train_gt_paths,
        'test_input_paths':test_input_paths,
        'test_gt_paths':test_gt_paths}
def snow(config):
    # import pdb;pdb.set_trace()
    snow_path = config.path.snow

    train_input_paths = os.path.join(snow_path,'train','input')
    train_gt_paths = os.path.join(snow_path,'train','gt')
    test_input_paths = os.path.join(snow_path,'test','input')
    test_gt_paths = os.path.join(snow_path,'test','gt')
    
    train_input_paths = glob_path(train_input_paths)
    train_gt_paths = glob_path(train_gt_paths)
    test_input_paths = glob_path(test_input_paths)
    test_gt_paths = glob_path(test_gt_paths)

    # check_path(train_input_paths, train_gt_paths)
    # check_path(test_input_paths, test_gt_paths)

    return {'train_input_paths':train_input_paths,
        'train_gt_paths':train_gt_paths,
        'test_input_paths':test_input_paths,
        'test_gt_paths':test_gt_paths}
def haze(config):
    # import pdb;pdb.set_trace()
    haze_path = config.path.haze

    train_input_paths = os.path.join(haze_path,'train','input')
    train_gt_paths = os.path.join(haze_path,'train','gt')
    test_input_paths = os.path.join(haze_path,'test','input')
    test_gt_paths = os.path.join(haze_path,'test','gt')
    
    train_input_paths = glob_path(train_input_paths)
    train_gt_paths = glob_path(train_gt_paths)
    test_input_paths = glob_path(test_input_paths)
    test_gt_paths = glob_path(test_gt_paths)

    # check_path(train_input_paths, train_gt_paths)
    # check_path(test_input_paths, test_gt_paths)

    return {'train_input_paths':train_input_paths,
        'train_gt_paths':train_gt_paths,
        'test_input_paths':test_input_paths,
        'test_gt_paths':test_gt_paths}


class AllLight:
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
                                          n=self.config.training.patch_n,
                                          patch_size=self.config.data.image_size,
                                          transforms=self.transforms,
                                          parse_patches=parse_patches)
        val_dataset = AllLightDataset(input_paths=test_input_paths,
                                        gt_paths=test_gt_paths,
                                        n=self.config.training.patch_n,
                                        patch_size=self.config.data.image_size,
                                        transforms=self.transforms,
                                        parse_patches=False)

        if not parse_patches:
            self.config.training.batch_size = 1
            self.config.sampling.batch_size = 1

        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=self.config.training.batch_size,
                                                   shuffle=True, num_workers=self.config.data.num_workers,
                                                   pin_memory=True)
        val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=1,
                                                 shuffle=False, num_workers=self.config.data.num_workers,
                                                 pin_memory=True)

        return train_loader, val_loader


class AllLightDataset(torch.utils.data.Dataset):
    def __init__(self, input_paths, gt_paths, patch_size, n, transforms, parse_patches=True):
        super().__init__()

        self.input_paths = input_paths
        self.gt_paths = gt_paths

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
        input_path = self.input_paths[index]
        gt_path = self.gt_paths[index]
        
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
        else:
            input_img = PIL.Image.open(input_path).convert('RGB')
            img_id = re.split('/', input_path)[-1][:-4]

            gt_img = PIL.Image.open(gt_path).convert('RGB')

        if self.parse_patches:
            i, j, h, w = self.get_params(input_img, (self.patch_size, self.patch_size), self.n)
            input_img = self.n_random_crops(input_img, i, j, h, w)
            gt_img = self.n_random_crops(gt_img, i, j, h, w)
            outputs = [torch.cat([self.transforms(input_img[i]), self.transforms(gt_img[i])], dim=0)
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

            return torch.cat([self.transforms(input_img), self.transforms(gt_img)], dim=0), img_id

    def __getitem__(self, index):
        res = self.get_images(index)
        return res

    def __len__(self):
        return len(self.input_paths)
