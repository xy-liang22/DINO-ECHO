# tasks_list=("LHF" "RHF" "DF" "LAD" "RAD" "LVD" "RVD" "AV_regurgitation" "AV_stenosis" "AV_vegetations" "MV_regurgitation" "MV_stenosis" "TV_regurgitation" "TV_regurgitation" "TV_vegetations" "PV_regurgitation" "PV_stenosis" "PV_vegetations" "PE" "LVH" "IMT" "IS")
tasks_list=("window" "view")
fold=("fold_4" "fold_3")
device=cuda:1
num_classes=(5 10)
dataset=view
for i in ${!tasks_list[@]}; do
    task=${tasks_list[i]}
    run_name=${task}_predict_full
    n_classes=${num_classes[i]}
    python run.py --model linear_classifier  \
                  --data_path /mnt/hanoverdev/scratch/hanwen/xyliang/ECHO_dataset_csv/label_view/${task}_predict_full.json \
                  --data_path_field path \
                  --dataset_csv /mnt/hanoverdev/scratch/hanwen/xyliang/ECHO_dataset_csv/label_view/${task}_predict_full.csv \
                  --dataclass EchoViewClassificationPredict \
                  --output_dir /mnt/hanoverdev/scratch/hanwen/xyliang/ECHO_results_view_predict/${run_name} \
                  --wandb_project ECHO \
                  --wandb_group view_classify \
                  --run_name ${run_name} \
                  --wandb_dir /mnt/hanoverdev/scratch/hanwen/xyliang/ECHO_results/wandb_log \
                  --hidden_dim 1024 \
                  --batch_size 512 \
                  --val_time 1 \
                  --epochs 100 \
                  --warmup_epochs 10 \
                  --num_workers 12 \
                  --num_classes ${n_classes} \
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
                  --resume /mnt/hanoverdev/scratch/hanwen/xyliang/ECHO_results_view/${task}_classify/${fold}/model_best.pth
done