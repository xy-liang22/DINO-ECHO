# tasks_list=("LHF" "RHF" "DF" "LAD" "RAD" "RVD" "AV_regurgitation" "AV_stenosis" "MV_regurgitation" "MV_stenosis" "TV_regurgitation" "PE" "LVH")
tasks_list=("PV_stenosis" "TV_stenosis")
device=cuda:2
num_classes=2
dataset=clip_study_only
# run_names=("study_original1_proj_fullsize" "study_original1_proj_fullsize_multi_videos" "clip" "study_proj" "study_proj_multi_videos" "study_transformer_proj" "study_transformer_original1_proj_fullsize")
# data_paths=("dinov2_study_original1_embeddings" "dinov2_study_original1_embeddings_multi_videos" "dinov2_clip_embeddings" "dinov2_study_embeddings" "dinov2_study_multi_videos_embeddings" "dinov2_transformer_embeddings" "dinov2_transformer_original1_fullsize_embeddings")
# pretrained_paths=("BiomedBERT_study_original1_proj_dist/checkpoints/epoch_20.pt" "BiomedBERT_study_original1_proj_dist/checkpoints/epoch_20.pt" "epoch_30_lr_3e-6_BiomedBERT/checkpoints/epoch_12.pt" "BiomedBERT_study_lr3e-6_frame80_bsz2_accum12_proj_dist/checkpoints/epoch_28.pt" "BiomedBERT_study_lr3e-6_frame80_bsz2_accum12_proj_dist/checkpoints/epoch_28.pt" "BiomedBERT_study_lr3e-6_frame80_bsz2_accum12_proj_transformer_dist/checkpoints/epoch_25.pt" "BiomedBERT_study_original1_proj_transformer_dist/checkpoints/epoch_21.pt")
# clip_models=("DINOv2_BiomedBERT_study_new" "DINOv2_BiomedBERT_study_new" "DINOv2_BiomedBERT" "DINOv2_BiomedBERT_study" "DINOv2_BiomedBERT_study" "DINOv2_BiomedBERT_study_transformer" "DINOv2_BiomedBERT_study_transformer_new")
run_names=("study_original1_proj_fullsize_multi_videos")
data_paths=("dinov2_study_original1_embeddings_multi_videos")
pretrained_paths=("BiomedBERT_study_original1_proj_dist/checkpoints/epoch_20.pt")
clip_models=("DINOv2_BiomedBERT_study_new")
# run_names=("study_proj_multi_videos")
# data_paths=("dinov2_study_multi_videos_embeddings")
# pretrained_paths=("BiomedBERT_study_lr3e-6_frame80_bsz2_accum12_proj_dist/checkpoints/epoch_28.pt")
# clip_models=("DINOv2_BiomedBERT_study")

for task in ${tasks_list[@]}; do
    for i in ${!run_names[@]}; do
        run_name_suffix=${run_names[i]}
        data_path=${data_paths[i]}
        run_name=${task}_clip_zeroshot_allvideos_${run_name_suffix}
        pretrained=${pretrained_paths[i]}
        clip_model_name=${clip_models[i]}
        python run.py --model clip_classifier  \
                    --clip_model_name ${clip_model_name} \
                    --data_path /data/ECHO/${data_path}_mean.pt \
                    --prompt_path /mnt/hanoverdev/scratch/hanwen/xyliang/Zero_shot_result/prompts.json \
                    --task_name ${task} \
                    --data_path_field path \
                    --dataset_csv /mnt/hanoverdev/scratch/hanwen/xyliang/ECHO_dataset_csv/label_dataset_v4_${dataset}/${task}.csv \
                    --dataclass EchoZeroShotClassification \
                    --output_dir /mnt/hanoverdev/scratch/hanwen/xyliang/Zero_shot_result/${run_name} \
                    --wandb_project ECHO \
                    --wandb_group zeroshot_label_v4 \
                    --run_name ${run_name} \
                    --wandb_dir /mnt/hanoverdev/scratch/hanwen/xyliang/ECHO_results/wandb_log \
                    --batch_size 512 \
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
                    --pretrained /mnt/hanoverdev/scratch/hanwen/xyliang/CLIP_logs/${pretrained} \
                    --n_bootstrap_eval 1000
    done
done
