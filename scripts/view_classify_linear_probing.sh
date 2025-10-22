# tasks_list=("LHF" "RHF" "DF" "LAD" "RAD" "LVD" "RVD" "AV_regurgitation" "AV_stenosis" "AV_vegetations" "MV_regurgitation" "MV_stenosis" "TV_regurgitation" "TV_regurgitation" "TV_vegetations" "PV_regurgitation" "PV_stenosis" "PV_vegetations" "PE" "LVH" "IMT" "IS")
# tasks_list=("window_test_study" "view_test_study")
tasks_list=("window" "view")
device=cuda:1
num_classes=(5 10)
dataset=view
data_paths=("dinov2_original1_fullsize_embeddings" "dinov2_public_embeddings")
run_names=("original1" "public")
hidden_dim=(1024 1024)
for i in ${!tasks_list[@]}; do
    task=${tasks_list[i]}
    n_classes=${num_classes[i]}
    for j in ${!run_names[@]}; do
        run_name_suffix=${run_names[j]}
        data_path=${data_paths[j]}
        run_name=${task}_classify_resplit_${run_name_suffix}
        hidden_dim_value=${hidden_dim[j]}
        python run.py --model linear_classifier  \
                    --task_name ${task} \
                    --data_path /data/ECHO/${data_path} \
                    --data_path_field path \
                    --dataset_csv /mnt/hanoverdev/scratch/hanwen/xyliang/ECHO_dataset_csv/label_${dataset}/${task}.csv \
                    --dataclass EchoViewClassification \
                    --output_dir /mnt/hanoverdev/scratch/hanwen/xyliang/ECHO_results_view/${run_name} \
                    --wandb_project ECHO \
                    --wandb_group view_classify \
                    --run_name ${run_name} \
                    --wandb_dir /mnt/hanoverdev/scratch/hanwen/xyliang/ECHO_results/wandb_log \
                    --hidden_dim ${hidden_dim_value} \
                    --batch_size 512 \
                    --val_time 1 \
                    --epochs 100 \
                    --warmup_epochs 5 \
                    --num_workers 12 \
                    --num_classes ${n_classes} \
                    --blr 5e-4 \
                    --layer_decay 0.95 \
                    --weight_decay 0.05 \
                    --dropout 0.1 \
                    --smoothing 0.0 \
                    --fold 5 \
                    --model_select val \
                    --balanced_dataset \
                    --device ${device} \
                    --save_freq 100
    done
done