python other/save_dinov2_slice_embeddings_multi_videos.py \
    --data_dir /mnt/hanoverdev/data/patxiao/ECHO_numpy/original_size/ \
    --pretrained /mnt/hanoverdev/scratch/hanwen/xyliang/DINOv2_study_original1.pt \
    --config_path models/dinov2_modules/configs/train/vitl16_lbsz96_500ep_resume_from_general.yaml \
    --embedding_dir /data/ECHO/dinov2_study_original1_slice_embeddings_multi_videos/ \
    --combine_embeddings \
    --image_size 224 \
    --batch_size 12 \
    --device cuda:3