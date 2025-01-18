version=$RESOLUTION ##1024, 512, 256
seed=$SEED
name=dynamicrafter_$RESOLUTION_seed${seed}

ckpt=checkpoints/dynamicrafter_$RESOLUTION_v1/model.ckpt
config=configs/inference_$RESOLUTION_v1.0.yaml

prompt_path=$PROMPT_PATH
res_dir=$OUTPUT_PATH
duration=40  # 5 seconds 91 sec per 
ddimsteps=70  # default 50 for shorter videos

if [ "$1" == "256" ]; then
    H=256
    FS=3  ## This model adopts frame stride=3, range recommended: 1-6 (larger value -> larger motion)
elif [ "$1" == "512" ]; then
    H=320
    # NOTE: derck changed this to 16 FPS
    FS=16 ## This model adopts FPS=16, range recommended: 15-30 (smaller value -> larger motion)
elif [ "$1" == "1024" ]; then
    H=576
    FS=10 ## This model adopts FPS=10, range recommended: 15-5 (smaller value -> larger motion)
else
    echo "Invalid input. Please enter 256, 512, or 1024."
    exit 1
fi


if [ "$1" == "256" ]; then
CUDA_VISIBLE_DEVICES=0 python3 scripts/evaluation/inference.py \
--seed ${seed} \
--ckpt_path $ckpt \
--config $config \
--savedir $res_dir/$name \
--n_samples 1 \
--bs 1 --height ${H} --width $1 \
--unconditional_guidance_scale 7.5 \
--ddim_steps ${ddimsteps} \
--ddim_eta 1.0 \
--prompt_path $prompt_path \
--text_input \
--video_length ${duration} \
--frame_stride ${FS}
else
CUDA_VISIBLE_DEVICES=0 python3 scripts/evaluation/inference.py \
--seed ${seed} \
--ckpt_path $ckpt \
--config $config \
--savedir $res_dir/$name \
--n_samples 1 \
--bs 1 --height ${H} --width $1 \
--unconditional_guidance_scale 7.5 \
--ddim_steps ${ddimsteps} \
--ddim_eta 1.0 \
--prompt_path $prompt_path \
--text_input \
--video_length ${duration} \
--frame_stride ${FS} \
--timestep_spacing 'uniform_trailing' --guidance_rescale 0.7 --perframe_ae
fi


## multi-cond CFG: the <unconditional_guidance_scale> is s_txt, <cfg_img> is s_img
#--multiple_cond_cfg --cfg_img 7.5
#--loop

