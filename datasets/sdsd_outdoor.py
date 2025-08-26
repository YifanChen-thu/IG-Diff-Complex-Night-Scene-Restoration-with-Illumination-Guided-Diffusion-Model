import os
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

class sdsd_outdoor:
    def __init__(self, config):
        self.config = config
        self.transforms = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])

    def get_loaders(self, parse_patches=True, validation='sdsd_outdoor'):
        if validation == 'raindrop':
            print("=> evaluating raindrop test set...")
            path = os.path.join(self.config.data.data_dir, 'data', 'raindrop', 'test')
            filename = 'raindroptesta.txt'
        elif validation == 'rainfog':
            print("=> evaluating outdoor rain-fog test set...")
            path = os.path.join(self.config.data.data_dir, 'data', 'outdoor-rain')
            filename = 'test1.txt'
        #lol
        elif validation == 'lol':
            print("=> evaluating outdoor lol test set...")
            path = os.path.join(self.config.data.data_dir, 'eval15')

        elif validation == 'lolv2_real':
            print("=> evaluating outdoor lolv2_real test set...")
            path = os.path.join(self.config.data.data_dir, 'Test')

        elif validation == 'lolv2_syn':
            print("=> evaluating outdoor lolv2_syn test set...")
            path = os.path.join(self.config.data.data_dir, 'Test')

        elif validation == 'sdsd_outdoor':
            print("=> evaluating sdsd_outdoor test set...")

            if self.config.data.testing_dir is not None:
                test_dir = self.config.data.testing_dir
                test_dir = test_dir.split(',')
            else:
                test_dir = []


            
            


        else:   # snow
            print("=> evaluating snowtest100K-L...")
            path = os.path.join(self.config.data.data_dir, 'data', 'snow100k')
            filename = 'snowtest100k_L.txt'

        # train_dataset = LOLDataset(os.path.join(self.config.data.data_dir,'our485'),
        #                                   n=self.config.training.patch_n,
        #                                   patch_size=self.config.data.image_size,
        #                                   transforms=self.transforms,
        #                                 #   filelist='allweather.txt',
        #                                   parse_patches=parse_patches)

        train_dataset = LOLDataset(self.config.data.data_dir,
                                       n=self.config.training.patch_n,
                                       patch_size=self.config.data.image_size,
                                       transforms=self.transforms,
                                       #   filelist='allweather.txt',
                                       parse_patches=parse_patches,
                                       test_dir=test_dir,test=False)

        val_dataset = LOLDataset(self.config.data.data_dir, n=self.config.training.patch_n,
                                        patch_size=self.config.data.image_size,
                                        transforms=self.transforms,
                                        # filelist=filename,
                                        parse_patches=parse_patches,
                                        test_dir=test_dir,test=True)

        if not parse_patches:
            self.config.training.batch_size = 1
            self.config.sampling.batch_size = 1

        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=self.config.training.batch_size,
                                                   shuffle=True, num_workers=self.config.data.num_workers,
                                                   pin_memory=True)
        val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=self.config.sampling.batch_size,
                                                 shuffle=False, num_workers=self.config.data.num_workers,
                                                 pin_memory=True)

        return train_loader, val_loader


class LOLDataset(torch.utils.data.Dataset):
    def __init__(self, dir, patch_size, n, transforms, filelist=None, parse_patches=True,test_dir = None,test=False):
        super().__init__()
        self.test = test

        self.dir = dir
        
        self.input_dir = os.path.join(dir,'input')
        self.gt_dir = os.path.join(dir,'GT')


        subfolders_input = util.glob_file_list(self.input_dir)
        subfolders_GT = util.glob_file_list(self.gt_dir)

        self.input_names_train = []
        self.gt_names_train = []

        self.input_names_test = []
        self.gt_names_test = []

        for subfolder_input,subfolder_gt in zip(subfolders_input,subfolders_GT):
            subfolder_name = os.path.basename(subfolder_gt)
            # input_names.append
            #train
            if not(subfolder_name in test_dir) and not(subfolder_name.split('_2')[0] in test_dir):
                img_paths_input_train = util.glob_file_list(subfolder_input)   #
                # print("img_paths_input_train:{}".format(img_paths_input_train))
                # print("size():{}".format(img_paths_input_train))
                img_paths_gt_train = util.glob_file_list(subfolder_gt)
                # print("img_paths_gt_train:{}".format(img_paths_gt_train))
                
                
                assert len(img_paths_input_train) == len(img_paths_gt_train)

                self.train_input_paths = img_paths_input_train
                self.train_gt_paths = img_paths_gt_train

                self.input_names_train += self.train_input_paths
                # print("self.input_name_train:{}".format(self.input_names_train))
                self.gt_names_train += self.train_gt_paths

                

            #test
            else:
                img_paths_input_test = util.glob_file_list(subfolder_input)
                img_paths_gt_test = util.glob_file_list(subfolder_gt)

                assert len(img_paths_input_test) == len(img_paths_gt_test)

                self.test_input_paths = img_paths_input_test
                self.test_gt_paths = img_paths_gt_test

                self.input_names_test += self.test_input_paths
                self.gt_names_test += self.test_gt_paths

        if test:
            self.input_names = self.input_names_test

        else:
            self.input_names = self.input_names_train
        # print(self.input_names[0])
        # self.input_names =  os.listdir(os.path.join(dir,'input'))
        # self.gt_names =  os.listdir(os.path.join(dir,'gt'))
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
        input_name = self.input_names[index]
        # print("input_name[-1]:{}".format(input_name[-1]))
        # print("input_name:{},self.input_names.shape:{}".format(input_name,self.input_names.shape))
        # gt_name = self.gt_names[index]
        if self.test :
            test_input_name_path = self.input_names_test[index]
            test_gt_name_path = self.gt_names_test[index]

            input_img = util.read_img3(test_input_name_path)
            gt_img = util.read_img3(test_gt_name_path)


        else:
            train_input_name_path = self.input_names_train[index]
            train_gt_name_path = self.gt_names_train[index]

            input_img = util.read_img3(train_input_name_path)
            gt_img = util.read_img3(train_gt_name_path)


        # input_img = PIL.Image.open(os.path.join(self.dir,input_name)) if self.dir else PIL.Image.open(input_name)
        # input_img = PIL.Image.open(os.path.join(self.dir, 'low',input_name)) if self.dir else PIL.Image.open(input_name)
        # input_img = PIL.Image.open(os.path.join(self.dir, 'input',input_name)) if self.dir else PIL.Image.open(input_name)
        # input_img_folder = PIL.Image.open(os.path.join(self.dir, 'input',input_folder_name)) if self.dir else PIL.Image.open(input_name)
            
        # try:
        #     # gt_img = PIL.Image.open(os.path.join(self.dir,gt_name)) if self.dir else PIL.Image.open(gt_name)
        #     # gt_img = PIL.Image.open(os.path.join(self.dir, 'high',gt_name)) if self.dir else PIL.Image.open(gt_name)
        #     gt_img = PIL.Image.open(os.path.join(self.dir, 'GT',gt_name)) if self.dir else PIL.Image.open(gt_name)

        # except:
        #     # gt_img = PIL.Image.open(os.path.join(self.dir,gt_name)).convert('RGB') if self.dir else PIL.Image.open(gt_name).convert('RGB')

        #     # gt_img = PIL.Image.open(os.path.join(self.dir, 'high',gt_name)).convert('RGB') if self.dir else PIL.Image.open(gt_name).convert('RGB')
        #     gt_img = PIL.Image.open(os.path.join(self.dir, 'GT',gt_name)).convert('RGB') if self.dir else PIL.Image.open(gt_name).convert('RGB')
        # print("input_names:{}".format(self.input_names))
        # print("inputname:{},inputname[-2]:{},inputname[-1]:{}".format(input_name,input_name[-2],input_name[-1]))
        # img_id = re.split('/', input_name)[-1][:-4]   
        # print(input_name)  
        img_id_1 = re.split('/',input_name)[-2]
        img_id_2 = re.split('/',input_name)[-1]
        img_id_2_1 = re.split('.npy',img_id_2)[0]
        # print(img_id_2_1)
        img_id = img_id_1 + '_' + img_id_2_1
        # print("img_id_1:{},img_id_2:{},img_id_2_1:{},img_id:{}".format(img_id_1,img_id_2,img_id_2_1,img_id))
    

        if self.parse_patches:
            i, j, h, w = self.get_params(input_img, (self.patch_size, self.patch_size), self.n)
            input_img = self.n_random_crops(input_img, i, j, h, w)
            gt_img = self.n_random_crops(gt_img, i, j, h, w)
            outputs = [torch.cat([self.transforms(input_img[i]), self.transforms(gt_img[i])], dim=0)
                       for i in range(self.n)]
            return torch.stack(outputs, dim=0), img_id
        else:
            # Resizing images to multiples of 16 for whole-image restoration
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
        return len(self.input_names)
