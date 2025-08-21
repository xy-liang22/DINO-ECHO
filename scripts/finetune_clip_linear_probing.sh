# tasks_list=("LHF" "RHF" "DF" "LAD" "RAD" "LVD" "RVD" "AV_regurgitation" "AV_stenosis" "AV_vegetations" "MV_regurgitation" "MV_stenosis" "TV_regurgitation" "TV_regurgitation" "TV_vegetations" "PV_regurgitation" "PV_stenosis" "PV_vegetations" "PE" "LVH" "IMT" "IS")
tasks_list=("surgery_indication")
device=cuda:1
num_classes=2
dataset=clip_study_only
for task in ${tasks_list[@]}; do
    run_name=${task}_clip_linear_allvideos_study_epoch100_new
    python run.py --model linear_classifier  \
                  --data_path /data/ECHO/dinov2_clip_embeddings_mean.pt \
                  --data_path_field path \
                  --dataset_csv /mnt/hanoverdev/scratch/hanwen/xyliang/ECHO_dataset_csv/label_dataset_v4_${dataset}/${task}.csv \
                  --dataclass EchoEmbeddingClassification \
                  --output_dir /mnt/hanoverdev/scratch/hanwen/xyliang/ECHO_results_${dataset}/${run_name} \
                  --wandb_project ECHO \
                  --wandb_group evaluate_label_v4 \
                  --run_name ${run_name} \
                  --wandb_dir /mnt/hanoverdev/scratch/hanwen/xyliang/ECHO_results/wandb_log \
                  --hidden_dim 1024 \
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
                  --save_freq 100
done