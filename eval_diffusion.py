import argparse
import os
import random
import socket
import yaml
import torch
import torch.backends.cudnn as cudnn
import numpy as np
import torchvision
import models
import datasets
import utils
from models import DenoisingDiffusion, DiffusiveRestoration
#python eval_diffusion.py --config "lol.yml" --resume '/home/cyf20/code/rrddln/WeatherDiffusion-main/ckpt/LOL_ddpm_100000.pth.tar' --test_set 'lol' --image_folder 'results/lol_test/' --sampling_timesteps 25 --grid_r 16

def parse_args_and_config():
    parser = argparse.ArgumentParser(description='Restoring Weather with Patch-Based Denoising Diffusion Models')
    parser.add_argument("--config", type=str, required=True,
                        help="Path to the config file")
    parser.add_argument('--resume', default='', type=str,
                        help='Path for the diffusion model checkpoint to load for evaluation')
    parser.add_argument("--grid_r", type=int, default=16,
                        help="Grid cell width r that defines the overlap between patches")
    parser.add_argument("--sampling_timesteps", type=int, default=25,
                        help="Number of implicit sampling steps")
    parser.add_argument("--test_set", type=str, default='raindrop',
                        help="restoration test set options: ['raindrop', 'snow', 'rainfog']")
    parser.add_argument("--test_input_path", type=str, default='raindrop')
    parser.add_argument("--test_input_sci_path", type=str, default='raindrop')
    parser.add_argument("--testing_block", type=str, default=None)
    parser.add_argument("--image_folder", default='results/images/', type=str,
                        help="Location to save restored images")
    parser.add_argument('--seed', default=61, type=int, metavar='N',
                        help='Seed for initializing training (default: 61)')
    args = parser.parse_args()

    with open(os.path.join("configs", args.config), "r") as f:
        config = yaml.safe_load(f)
    new_config = dict2namespace(config)

    new_config.data.data_name = args.test_set
    new_config.path.real_weather = args.test_input_path
    new_config.path.real_weather_sci = args.test_input_sci_path

    if args.testing_block is not None:
        new_config.data.testing_block = args.testing_block
    return args, new_config


def dict2namespace(config):
    namespace = argparse.Namespace()
    for key, value in config.items():
        if isinstance(value, dict):
            new_value = dict2namespace(value)
        else:
            new_value = value
        setattr(namespace, key, new_value)
    return namespace


def main():
    args, config = parse_args_and_config()

    # setup device to run
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    print("Using device: {}".format(device))
    config.device = device

    if torch.cuda.is_available():
        print('Note: Currently supports evaluations (restoration) when run only on a single GPU!')

    # set random seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.benchmark = True

    # data loading
    print("=> using dataset '{}'".format(config.data.dataset))
    DATASET = datasets.__dict__[config.data.dataset](config)
    _, val_loader = DATASET.get_loaders(parse_patches=False, validation=args.test_set)

    # create model
    print("=> creating denoising-diffusion model with wrapper...")
    diffusion = DenoisingDiffusion(args, config)

    if os.path.isfile(args.resume):
        diffusion.load_ddm_ckpt(args.resume, ema=True)
        diffusion.model.eval()
    else:
        print('Pre-trained diffusion model path is missing!')
        raise NotImplementedError
    
    model = DiffusiveRestoration(diffusion, args, config)
    image_folder = os.path.join(args.image_folder)
    os.makedirs(image_folder, exist_ok=True)

    cnt = 0
    for p in model.diffusion.model.parameters():
        cnt += p.nelement()
    print(f'Parameters: {cnt}')
    # exit()
    args.grid_r = config.data.image_size // 2 if args.grid_r is None else args.grid_r
    
    import time
    time_start = time.time()

    model.restore(image_folder, val_loader, r=args.grid_r, max_image_number=None, verbose=True)
    
    time_end = time.time()

    print(f'test time:{time_end-time_start}')

if __name__ == '__main__':
    main()



#

#