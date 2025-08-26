import os
import cv2
from utils.metrics import calculate_psnr, calculate_ssim

#gt 
# mode = 'lol_v1'
# gt_root = '../../data/LOL-v1/eval15/high'
# results_root_list = [
#     'final_results/lol_v1/lol_v1_cond_mini_222w',
#     # '/apdcephfs_cq2/share_1290939/feiiyin/Lowlevel/low-light-diffusion/results/lolv1_unetcondition_illu_256_real_test//LOL_guided/lol/',
#     # '/apdcephfs_cq2/share_1290939/feiiyin/Lowlevel/low-light-diffusion/results/lolv1_unetcondition_illu_kv_1-minus-illu_test//LOL_guided/lol/',
#     # '/apdcephfs_cq2/share_1290939/feiiyin/Lowlevel/low-light-diffusion/results/lolv1_unetcondition_illu_kv_test//LOL_guided/lol/',
#     # '/apdcephfs_cq2/share_1290939/feiiyin/Lowlevel/low-light-diffusion/results/lolv1_unetcondition_refl_test//LOL_guided/lol/',
#     # '/apdcephfs_cq2/share_1290939/feiiyin/Lowlevel/low-light-diffusion/results/lolv1_learnable_q_256_test/LOL_guided/LOL_guided/'
#     # '/apdcephfs_cq2/share_1290939/feiiyin/Lowlevel/low-light-diffusion/results/lolv1_learnable_q_64_test/LOL_guided/LOL_guided/'
#     # '/apdcephfs_cq2/share_1290939/feiiyin/Lowlevel/low-light-diffusion/results/lolv1_sci_test/',
#     # '/apdcephfs_cq2/share_1290939/feiiyin/Lowlevel/low-light-diffusion/results/lolv1_sci_illu_fpn_test/LOL_guided/LOL_guided/',
#     # 'test_results/lol_v1/lol_v1_lol_v2_baseline_100w/',
#     # 'test_results/lol_v1/lol_v1_lol_v2_sdsd_128_baseline_100w/',
#     # 'test_results/lol_v1/lol_v1_lol_v2_sdsd_baseline_100w/',
# ]

# mode = 'lol_syn'
# gt_root = '../../data/LOL-v2/Synthetic/Test/Normal/'
# results_root_list = [
#     # 'final_results/lol_v2_syn/lol_v2_syn_cond_144w'
#     'final_results/lol_v2_syn/lol_v2_syn_cond_mini_222w'
# ]

# # # lol_real
# mode = 'lol_v2_real'
# gt_root = '../../data/LOL-v2/Real_captured/Test/Normal/'
# results_root_list = [
#     'final_results/lol_v2_real/lol_v2_real_cond_mini_222w'
# ]

# # rain
# mode = 'rain'
# gt_root = '../../data/DDN/test/gt/'
# results_root_list = [
#     # 'results/rain'
#     'final_results/rain/rain_cond_448w'
# ]
# raindrop
# mode = 'raindrop'
# gt_root = '../../data/Raindrop/test/gt/'
# results_root_list = [
#     # 'results/raindrop'
#     'final_results/raindrop/raindrop_cond_470w'
# ]
# # snow
# mode = 'snow'
# gt_root = '../../data/CityscapeSnow/test/gt/'
# results_root_list = [
#     # 'final_results/snow/snow_cond_134w'
#     'final_results/snow/snow_cond_490w'
# ]
# snow11k
mode = 'snow11k'
gt_root = '../../data/snow11k/test/gt/'
results_root_list = [
    # 'final_results/snow/snow_cond_134w'
    # 'final_results/snow11k/snow11k_cond_38w',
    # 'final_results/snow11k/snow11k_cond_68w',
    'final_results/snow11k/snow11k_cond_92w'
]

# # haze
# mode = 'haze'
# gt_root = '../../data/Haze4K/test/gt/'
# results_root_list = [
#     'final_results/haze/haze_cond_100w',
#     'final_results/haze/haze_cond_202w'
# ]
# fog
# mode = 'fog'
# gt_root = '../../data/CityscapeFog/test/gt/'
# results_root_list = [
#     # 'final_results/fog/fog_cond_114w',
#     # 'final_results/fog/fog_cond_132w',
#     # 'final_results/fog/fog_cond_414w',
#     # 'final_results/fog/fog_cond_424w',
#     'final_results/fog/fog_cond_600w'
# ]
#lol_blur

#snow11k



def name_mapper(gt_name, mode):
    if mode == 'lol_v2_real':
        img_name = gt_name.replace('normal', 'low') + '_output.png'
    elif mode == 'lol_blur':
        img_name = f'{gt_name}.png'
    elif mode == 'snow':
        img_name = f'{gt_name}.png'
    elif mode == 'rain':
        img_name = f'{gt_name}.png'
    # elif mode == 'haze':
    #     img_name = f'{gt_name}.png'
    elif mode == 'raindrop':
        img_name = gt_name.replace('clean', 'rain') + '_output.png'
    elif mode == 'fog':
        img_name = gt_name.replace('clean', 'hazy') + '_output.png'

    else:
        img_name = f'{gt_name}_output.png'
    return img_name


for results_root in results_root_list:
    print(f'[LOG]: {results_root}')

    imgsName = sorted(os.listdir(results_root))
    gtsName = sorted(os.listdir(gt_root))
    # print(imgsName,gtsName)

    # print(f'len(imgsName):{len(imgsName)},len(gtsName):{len(gtsName)}')
    assert len(imgsName) == len(gtsName), f'len(imgsName):{len(imgsName)},len(gtsName):{len(gtsName)}'

    cumulative_psnr, cumulative_ssim = 0, 0
    for i in range(len(gtsName)):
        gt_name = os.path.basename(gtsName[i])
        res_id = gt_name.split('.')[0]
        gt_path = os.path.join(gt_root, gt_name)
        
        img_name = name_mapper(res_id, mode=mode)
        img_path = os.path.join(results_root, img_name)

        assert os.path.exists(img_path), f'img_path: {img_path} not exist'
        assert os.path.exists(gt_path), f'gt_path: {gt_path} not exist'
        
        
        res = cv2.imread(img_path, cv2.IMREAD_COLOR)
        gt = cv2.imread(gt_path, cv2.IMREAD_COLOR)
        
        W, H, _ = res.shape
        gt = cv2.resize(gt, (H, W), interpolation = cv2.INTER_CUBIC)
        

        cur_psnr = calculate_psnr(res, gt, test_y_channel=True)
        cur_ssim = calculate_ssim(res, gt, test_y_channel=True)
        # print('PSNR is %.4f and SSIM is %.4f' % (cur_psnr, cur_ssim))
        cumulative_psnr += cur_psnr
        cumulative_ssim += cur_ssim
    val_psnr = cumulative_psnr / len(imgsName)
    val_ssim = cumulative_ssim / len(imgsName)
    print('Testing set, PSNR is %.4f and SSIM is %.4f' % (val_psnr, val_ssim))
    iters_num = results_root.split('_')[-1]
    log_file_name = os.path.join('results_log', mode, f'{mode}_{iters_num}.txt')
    if os.path.exists('results_log') == False:
        os.mkdir('results_log')
    if os.path.exists(os.path.join('results_log', mode)) == False:
        os.mkdir(os.path.join('results_log', mode)) 

    with open(log_file_name, 'a') as f:
        f.write(f"dataset_input: {results_root} \n")
        f.write("Overall: PSNR {:4f} SSIM {:4f} \n".format(val_psnr, val_ssim))
