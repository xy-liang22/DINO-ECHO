accum=12
bsz=2
frame=10000
lr=3e-6
frameratio=1
model=hf-hub:mkaichristensen/echo-clip
name=EchoCLIP_study_frame${frame}_test
python -m open_clip_train.main \
    --logs-dir /mnt/hanoverdev/scratch/hanwen/xyliang/CLIP_test \
    --wandb-project-name "DINOv2_CLIP" \
    --name ${name} \
    --save-frequency 1 \
    --log-every-n-steps 1 \
    --save-most-recent \
    --report-to wandb \
    --dataset-type custom_study \
    --val-data="/mnt/hanoverdev/scratch/hanwen/xyliang/CLIP_dataset_csv/clip_study/test_report.json"  \
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
    --model ${model} \
    --device cuda:1 \
    --precision bf16