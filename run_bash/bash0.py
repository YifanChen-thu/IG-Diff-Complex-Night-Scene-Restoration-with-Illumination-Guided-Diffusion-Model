# author: muzhan
# contact: levio.pku@gmail.com
import os
import sys
import time
 
cmd = 'OMP_THREADS_NUM=8 CUDA_VISIBLE_DEVICES=0 python eval_diffusion.py --config "all_light_condition.yml" --resume ckpt/snow11k/AllLightCondition_ddpm3360000.pth.tar  --test_set real_weather --image_folder final_results/real_weather/real_snow_snow11k_cond_336w/ --sampling_timesteps 25 --grid_r 16 --testing_block 1/2'
# cmd = 'CUDA_VISIBLE_DEVICES=0 ./train.sh Motion_Deblurring/Options/pca/pca_md_block_2_8.yml'

 
def gpu_info():
    gpu_status = os.popen('nvidia-smi | grep %').read().split('|')
    gpu_memory = int(gpu_status[2].split('/')[0].split('M')[0].strip())
    gpu_power = int(gpu_status[1].split('   ')[-1].split('/')[0].split('W')[0].strip())
    return gpu_power, gpu_memory
 
 
def narrow_setup(interval=2):
    gpu_power, gpu_memory = gpu_info()
    i = 0
    while gpu_memory > 500 : #or gpu_power > 200:  # set waiting condition
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
