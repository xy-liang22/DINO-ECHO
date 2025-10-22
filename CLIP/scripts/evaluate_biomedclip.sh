accum=12
bsz=1
frame=10000
lr=3e-6
frameratio=1
model=hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224
name=BiomedCLIP_study_frame${frame}_original_test
python -m open_clip_train.main \
    --logs-dir /mnt/hanoverdev/scratch/hanwen/xyliang/CLIP_test \
    --wandb-project-name "DINOv2_CLIP" \
    --name ${name} \
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
    --model ${model} \
    --device cuda:0 \
    --precision bf16