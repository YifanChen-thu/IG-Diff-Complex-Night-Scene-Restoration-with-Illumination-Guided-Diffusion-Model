ROOT=/apdcephfs_cq2/share_1290939/feiiyin/Lowlevel/low-light-diffusion/ckpt/
NAME=lol_v1_lol_v2_sdsd_baseline
CKPT_PATH=${ROOT}/${NAME}/AllLight_ddpm1000000.pth.tar
for TEST_SET in sdsd_outdoor
do
echo test_results/${TEST_SET}/${NAME}_100w/
CUDA_VISIBLE_DEVICES=3 python eval_diffusion.py --config "all_light.yml" \
 --resume ${CKPT_PATH} \
 --test_set ${TEST_SET} \
 --image_folder test_results/${TEST_SET}/${NAME}_100w/ --sampling_timesteps 25 --grid_r 16
done