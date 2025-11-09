tasks_list=("LHF" "RHF" "DF" "LAD" "RAD" "LVD" "RVD" "AV_regurgitation" "AV_stenosis" "MV_regurgitation" "MV_stenosis" "TV_regurgitation" "TV_stenosis" "PV_regurgitation" "PV_stenosis" "PE" "LVH")
fold=("fold_3" "fold_3" "fold_1" "fold_4" "fold_3" "fold_0" "fold_2" "fold_3" "fold_1" "fold_0" "fold_3" "fold_2" "fold_1" "fold_1" "fold_3" "fold_2" "fold_2")
device=cuda:1
for i in ${!tasks_list[@]}; do
    task=${tasks_list[i]}
    run_name=${task}_predict
    python run.py --model linear_classifier  \
                  --data_path /data/ECHO/dinov2_study_original1_embeddings_multi_videos_mean.pt \
                  --data_path_field path \
                  --dataset_csv /data/ECHO/llava_data_label/task_predict.csv \
                  --dataclass EchoEmbeddingClassificationPredict \
                  --output_dir /mnt/hanoverdev/scratch/hanwen/xyliang/ECHO_results_task_predict/${run_name} \
                  --wandb_project ECHO \
                  --wandb_group task_predict \
                  --run_name ${run_name} \
                  --wandb_dir /mnt/hanoverdev/scratch/hanwen/xyliang/ECHO_results/wandb_log \
                  --hidden_dim 1024 \
                  --batch_size 512 \
                  --val_time 1 \
                  --epochs 100 \
                  --warmup_epochs 10 \
                  --num_workers 12 \
                  --num_classes 2 \
                  --blr 5e-4 \
                  --layer_decay 0.95 \
                  --weight_decay 0.05 \
                  --dropout 0.1 \
                  --smoothing 0.0 \
                  --fold 5 \
                  --model_select val \
                  --device ${device} \
                  --save_freq 100 \
                  --predict \
                  --resume /mnt/hanoverdev/scratch/hanwen/xyliang/ECHO_results_clip_study_only/${task}_clip_linear_allvideos_study_original1_multi_videos_fullsize_epoch100_new/${fold}/model_best.pth
done