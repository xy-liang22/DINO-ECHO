task=LHF
num_classes=2
python run.py --model dinov2_large_classifier \
                        --data_path /mnt/hanoverdev/data/patxiao/ECHO_numpy/20250126/ \
                        --data_path_field path \
                        --dataset_csv /mnt/hanoverdev/scratch/hanwen/xyliang/ECHO_dataset_csv/label_dataset_v4/${task}.csv \
                        --dataclass EchoData \
                        --eval \
                        --eval_path /mnt/hanoverdev/scratch/hanwen/xyliang/ECHO_results/out_${task}_v4/fold_0/result_best_model.json \
                        --batch_size 4 \
                        --image_size 256 \
                        --num_frames 64 \
                        --t_patch_size 8 \
                        --max_frames 128 \
                        --num_workers 20 \
                        --num_classes ${num_classes} \
                        --smoothing 0.0 \
                        --fold 1 \
                        --pretrained /mnt/hanoverdev/scratch/hanwen/exp/echofound/pretrain_dinov2/20250205_vitl16_lbsz64_gbsz512_500ep_noKoleo/eval/training_624999/teacher_checkpoint.pth \
                        --resume /mnt/hanoverdev/scratch/hanwen/xyliang/ECHO_results/out_${task}_v4/fold_0/model_best.pth \
                        --device cuda:2