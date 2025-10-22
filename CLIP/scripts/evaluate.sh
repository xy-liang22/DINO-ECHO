bsz=1
frame=10000
lr=3e-6
frameratio=1
python -m open_clip_train.main \
    --logs-dir /mnt/hanoverdev/scratch/hanwen/xyliang/CLIP_logs \
    --wandb-project-name "DINOv2_CLIP" \
    --name "BiomedBERT_study_original1_proj_dist_frame${frame}_test"\
    --save-frequency 1 \
    --log-every-n-steps 1 \
    --save-most-recent \
    --report-to wandb \
    --dataset-type custom_study \
    --val-data="/mnt/hanoverdev/scratch/hanwen/xyliang/CLIP_dataset_csv/clip_study/test_report_original.json"  \
    --csv-separator "," \
    --csv-img-key study \
    --csv-caption-key report \
    --batch-size ${bsz} \
    --wd=0.1 \
    --epochs=1 \
    --workers=16 \
    --video-max-frames ${frame} \
    --video-frames-ratio ${frameratio} \
    --num-videos 4 \
    --video-interpolation="bilinear" \
    --model DINOv2_BiomedBERT_study_new \
    --pretrained /mnt/hanoverdev/scratch/hanwen/xyliang/CLIP_logs/BiomedBERT_study_original1_proj_dist/checkpoints/epoch_20.pt \
    --device cuda:0