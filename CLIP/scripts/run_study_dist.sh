export WORLD_SIZE=4
export CUDA_VISIBLE_DEVICES=0,1,2,3
accum=12
bsz=2
frame=80
lr=3e-6
frameratio=0.5
torchrun --nproc_per_node 4 -m open_clip_train.main \
    --logs-dir /mnt/hanoverdev/scratch/hanwen/xyliang/CLIP_logs \
    --wandb-project-name "DINOv2_CLIP" \
    --name "BiomedBERT_study_original1_proj_dist"\
    --save-frequency 1 \
    --accum-freq ${accum} \
    --log-every-n-steps 1 \
    --save-most-recent \
    --report-to wandb \
    --dataset-type custom_study \
    --train-data="/mnt/hanoverdev/scratch/hanwen/xyliang/CLIP_dataset_csv/clip_study/train_report_original.json"  \
    --val-data="/mnt/hanoverdev/scratch/hanwen/xyliang/CLIP_dataset_csv/clip_study/val_report_original.json"  \
    --csv-separator "," \
    --csv-img-key study \
    --csv-caption-key report \
    --warmup 442 \
    --batch-size ${bsz} \
    --lr=${lr} \
    --wd=0.1 \
    --epochs=30 \
    --workers=16 \
    --video-max-frames ${frame} \
    --video-frames-ratio ${frameratio} \
    --num-videos 4 \
    --video-interpolation="bilinear" \
    --model DINOv2_BiomedBERT_study_new \
    --dist-backend nccl \
    --gather-with-grad \
    --local-loss \
    --loss-dist-impl gather