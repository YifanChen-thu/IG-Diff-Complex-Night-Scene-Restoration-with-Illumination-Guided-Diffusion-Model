# author: muzhan
# contact: levio.pku@gmail.com
import os
import sys
import time
 
cmd = 'OMP_THREADS_NUM=8 CUDA_VISIBLE_DEVICES=4 python eval_diffusion.py --config "all_light_condition.yml" --resume ckpt/lol_blur_cond/AllLightCondition_ddpm3720000.pth.tar  --test_set lol_blur --image_folder final_results/lol_blur/lol_blur_cond_372w/ --sampling_timesteps 25 --grid_r 16 --testing_block 2/4'

 
def gpu_info():
    gpu_status = os.popen('nvidia-smi | grep %').read().split('|')
    gpu_memory = int(gpu_status[18].split('/')[0].split('M')[0].strip())
    gpu_power = int(gpu_status[17].split('   ')[-1].split('/')[0].split('W')[0].strip())
    return gpu_power, gpu_memory
 
 
def narrow_setup(interval=2):
    gpu_power, gpu_memory = gpu_info()
    i = 0
    while gpu_memory > 260 : #or gpu_power > 200:  # set waiting condition
        gpu_power, gpu_memory = gpu_info()
        i = i % 5
        symbol = 'monitoring: ' + '>' * i + ' ' * (10 - i - 1) + '|'
        # gpu_power_str = 'gpu power:%d W |' % gpu_power
        gpu_memory_str = 'gpu memory:%d MiB |' % gpu_memory
        sys.stdout.write('\r' + gpu_memory_str +  ' ' + symbol)  #' ' + gpu_power_str +
        sys.stdout.flush()
        time.sleep(interval)
        i += 1
    print('\n' + cmd)
    os.system(cmd)
 
 
if __name__ == '__main__':
    narrow_setup()
