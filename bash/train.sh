
CUDA_VISIBLE_DEVICES=0,1,2,3 \
python train_diffusion.py --config "all_light_condition.yml" \
--training_name=lol_v1_cond \
--data_name=lol_v1 \
--batch_size=16 \
--resume=ckpt/lol_v1_cond/AllLightCondition_ddpm40000.pth.tar

CUDA_VISIBLE_DEVICES=0,1,2,3 \
python train_diffusion.py --config "all_light_condition.yml" \
--training_name=lol_v2_real_cond \
--data_name=lol_v2_real \
--batch_size=16 \
--resume=ckpt/lol_v2_real_cond/AllLightCondition_ddpm40000.pth.tar

CUDA_VISIBLE_DEVICES=4,5,6,7 \
python train_diffusion.py --config "all_light_condition.yml" \
--training_name=lol_v2_syn_cond \
--data_name=lol_v2_syn \
--batch_size=16 \
--resume=ckpt/lol_v2_syn_cond/AllLightCondition_ddpm60000.pth.tar

CUDA_VISIBLE_DEVICES=0,1,2,3 \
python train_diffusion.py --config "all_light_condition.yml" \
--training_name=sdsd_indoor_cond \
--data_name=sdsd_indoor \
--batch_size=16 \
--resume=ckpt/sdsd_indoor_cond/AllLightCondition_ddpm40000.pth.tar

CUDA_VISIBLE_DEVICES=4,5,6,7 \
python train_diffusion.py --config "all_light_condition.yml" \
--training_name=sdsd_outdoor_cond \
--data_name=sdsd_outdoor \
--batch_size=16 \
--resume=ckpt/sdsd_outdoor_cond/AllLightCondition_ddpm40000.pth.tar

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
python train_diffusion.py --config "all_light_condition.yml" \
--training_name=lol_v1_lol_v2_sdsd_cond \
--data_name=lol_v1,lol_v2_real,lol_v2_syn,sdsd_indoor,sdsd_outdoor \
--batch_size=32 \
--resume=ckpt/lol_v1_lol_v2_sdsd_cond/AllLightCondition_ddpm40000.pth.tar