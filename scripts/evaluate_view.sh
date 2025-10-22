# tasks_list=("LHF" "RHF" "DF" "LAD" "LVD" "RAD" "RVD" "AV_regurgitation" "AV_stenosis" "MV_regurgitation" "MV_stenosis" "TV_regurgitation" "PV_regurgitation" "PE" "LVH")
tasks_list=("window" "view")
device=cuda:3
num_classes=(5 10)
dataset=view
# run_names=("study_multi_videos_epoch100_new" "original_epoch100_new" "public_epoch100_new" "echoclip_epoch100")
# data_paths=("dinov2_study_multi_videos_embeddings" "dinov2_original_embeddings" "dinov2_public_embeddings" "echoclip_embeddings")
# hidden_dim=(1024 1024 1024 512)
run_names=("classify_resplit" "classify_resplit_echoclip")
data_paths=("dinov2_study_original1_embeddings_multi_videos" "echoclip_embeddings")
hidden_dim=(1024 512)
# run_names=("study_original1_multi_videos_fullsize_epoch100_new")
# data_paths=("dinov2_study_original1_embeddings_multi_videos")
# hidden_dim=(1024)
for j in ${!tasks_list[@]}; do
    task=${tasks_list[j]}
    num_classes_value=${num_classes[j]}
    for i in ${!run_names[@]}; do
        run_name_suffix=${run_names[i]}
        data_path=${data_paths[i]}
        run_name=${task}_${run_name_suffix} 
        hidden_dim_value=${hidden_dim[i]}
        python run.py --model linear_classifier  \
                    --data_path /data/ECHO/${data_path} \
                    --data_path_field path \
                    --dataset_csv /mnt/hanoverdev/scratch/hanwen/xyliang/ECHO_dataset_csv/label_${dataset}/${task}.csv \
                    --dataclass EchoViewClassification \
                    --output_dir /mnt/hanoverdev/scratch/hanwen/xyliang/ECHO_results_${dataset}/${run_name} \
                    --wandb_project ECHO \
                    --wandb_group view_classify \
                    --run_name ${run_name} \
                    --wandb_dir /mnt/hanoverdev/scratch/hanwen/xyliang/ECHO_results/wandb_log \
                    --hidden_dim ${hidden_dim_value} \
                    --batch_size 512 \
                    --val_time 1 \
                    --epochs 100 \
                    --warmup_epochs 2 \
                    --num_workers 12 \
                    --num_classes ${num_classes_value} \
                    --blr 5e-4 \
                    --layer_decay 0.95 \
                    --weight_decay 0.05 \
                    --dropout 0.1 \
                    --smoothing 0.0 \
                    --fold 5 \
                    --model_select val \
                    --balanced_dataset \
                    --device ${device} \
                    --save_freq 100 \
                    --eval \
                    --resume /mnt/hanoverdev/scratch/hanwen/xyliang/ECHO_results_${dataset}/${run_name}/fold_0/model_best.pth \
                    --n_bootstrap_eval 1000
    done
done
