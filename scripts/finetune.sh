task=RHF
num_classes=2
python run.py --model dinov2_large_classifier \
                        --data_path /mnt/hanoverdev/data/patxiao/ECHO_numpy/20250126/ \
                        --data_path_field path \
                        --dataset_csv /mnt/hanoverdev/scratch/hanwen/xyliang/ECHO_dataset_csv/label_dataset_v4/${task}.csv \
                        --dataclass EchoData \
                        --output_dir /mnt/hanoverdev/scratch/hanwen/xyliang/ECHO_results/${task}_v4+2 \
                        --wandb_project ECHO \
                        --wandb_group evaluate_label_v4 \
                        --run_name ${task}_v4+2 \
                        --wandb_dir /mnt/hanoverdev/scratch/hanwen/xyliang/ECHO_results/wandb_log \
                        --batch_size 4 \
                        --val_time 4 \
                        --image_size 256 \
                        --num_frames 64 \
                        --t_patch_size 8 \
                        --epochs 5 \
                        --warmup_epochs 1 \
                        --max_frames 128 \
                        --num_workers 12 \
                        --num_classes ${num_classes} \
                        --blr 1e-3 \
                        --layer_decay 0.95 \
                        --weight_decay 0.05 \
                        --dropout 0.1 \
                        --smoothing 0.0 \
                        --fold 3 \
                        --pretrained /mnt/hanoverdev/scratch/hanwen/exp/echofound/pretrain_dinov2/20250205_vitl16_lbsz64_gbsz512_500ep_noKoleo/eval/training_624999/teacher_checkpoint.pth \
                        --model_select val \
                        --balanced_dataset \
                        --device cuda:1 \
                        