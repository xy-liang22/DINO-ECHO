# tasks_list=("LHF" "RHF" "DF" "LAD" "RAD" "LVD" "RVD" "AV_regurgitation" "AV_stenosis" "AV_vegetations" "MV_regurgitation" "MV_stenosis" "TV_regurgitation" "TV_regurgitation" "TV_vegetations" "PV_regurgitation" "PV_stenosis" "PV_vegetations" "PE" "LVH" "IMT" "IS")
# tasks_list=("LHF" "RHF" "DF" "LAD" "RAD" "LVD" "RVD" "AV_regurgitation" "AV_stenosis" "MV_regurgitation" "MV_stenosis" "TV_regurgitation" "PV_regurgitation" "PE" "LVH")
# tasks_list=("TV_stenosis" "PV_stenosis")
# tasks_list=("LVH" "PE")
tasks_list=("surgery_indication")
device=cuda:1
num_classes=2
dataset=clip_study_only
# run_names=("study_multi_videos_epoch100_new_crossval" "original_epoch100_new_crossval" "public_epoch100_new_crossval" "echoclip_epoch100_crossval")
run_names=("study_original1_multi_videos_fullsize_epoch100_new_crossval" "original1_fullsize_epoch100_new_crossval" "study_multi_videos_epoch100_new_crossval" "original_epoch100_new_crossval" "public_epoch100_new_crossval" "echoclip_epoch100_crossval")
# data_paths=("dinov2_study_multi_videos_embeddings" "dinov2_original_embeddings" "dinov2_public_embeddings" "echoclip_embeddings")
data_paths=("dinov2_study_original1_embeddings_multi_videos" "dinov2_original1_fullsize_embeddings" "dinov2_study_multi_videos_embeddings" "dinov2_original_embeddings" "dinov2_public_embeddings" "echoclip_embeddings")
# hidden_dim=(1024 1024 1024 512)
hidden_dim=(1024 1024 1024 1024 1024 512)
for task in ${tasks_list[@]}; do
    for i in ${!run_names[@]}; do
        run_name_suffix=${run_names[i]}
        data_path=${data_paths[i]}
        run_name=${task}_clip_linear_allvideos_${run_name_suffix}
        hidden_dim_value=${hidden_dim[i]}
        python run.py --model linear_classifier  \
                    --task_name ${task} \
                    --data_path /data/ECHO/${data_path}_mean.pt \
                    --data_path_field path \
                    --dataset_csv /mnt/hanoverdev/scratch/hanwen/xyliang/ECHO_dataset_csv/label_dataset_v4_${dataset}/${task}.csv \
                    --dataclass EchoEmbeddingClassification \
                    --output_dir /mnt/hanoverdev/scratch/hanwen/xyliang/ECHO_results_${dataset}/${run_name} \
                    --wandb_project ECHO \
                    --wandb_group evaluate_label_v4 \
                    --run_name ${run_name} \
                    --wandb_dir /mnt/hanoverdev/scratch/hanwen/xyliang/ECHO_results/wandb_log \
                    --hidden_dim ${hidden_dim_value} \
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
                    --choose_mini
    done
done