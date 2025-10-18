export CUDA_LAUNCH_BLOCKING=1
export TORCH_USE_CUDA_DSA=1

# tasks_list=("LHF" "RHF" "DF" "LAD" "LVD" "RAD" "RVD" "AV_regurgitation" "AV_stenosis" "MV_regurgitation" "MV_stenosis" "PV_regurgitation" "TV_regurgitation" "PE" "LVH")
tasks_list=("TV_stenosis" "PV_stenosis")
device=cuda:2
num_classes=2
dataset=clip_study_only
run_names=("biomedgpt")
data_paths=("/mnt/hanoverdev/data/patxiao/ECHO_numpy/20250126/")

for task in ${tasks_list[@]}; do
    for i in ${!run_names[@]}; do
        run_name_suffix=${run_names[i]}
        data_path=${data_paths[i]}
        run_name=${task}_clip_zeroshot_allvideos_${run_name_suffix}
        pretrained=${pretrained_paths[i]}
        python run.py --model biomedgpt_classifier  \
                    --task_name ${task} \
                    --data_path /mnt/hanoverdev/data/patxiao/ECHO_numpy/original_size \
                    --prompt_path /mnt/hanoverdev/scratch/hanwen/xyliang/Zero_shot_result/prompts_biomedgpt.json \
                    --task_name ${task} \
                    --data_path_field path \
                    --dataset_csv /mnt/hanoverdev/scratch/hanwen/xyliang/ECHO_dataset_csv/label_dataset_v4_${dataset}/${task}.csv \
                    --dataclass EchoBiomedGPTClassification \
                    --output_dir /mnt/hanoverdev/scratch/hanwen/xyliang/Zero_shot_result/${run_name} \
                    --wandb_project ECHO \
                    --wandb_group zeroshot_label_v4 \
                    --run_name ${run_name} \
                    --wandb_dir /mnt/hanoverdev/scratch/hanwen/xyliang/ECHO_results/wandb_log \
                    --batch_size 16 \
                    --val_time 1 \
                    --epochs 100 \
                    --warmup_epochs 2 \
                    --num_workers 12 \
                    --num_classes ${num_classes} \
                    --blr 5e-4 \
                    --layer_decay 0.95 \
                    --weight_decay 0.05 \
                    --dropout 0.1 \
                    --smoothing 0.0 \
                    --fold 5 \
                    --model_select auroc \
                    --balanced_dataset \
                    --device ${device} \
                    --save_freq 100 \
                    --eval \
                    --n_bootstrap_eval 1000
    done
done
