python other/save_dinov2_slice_embeddings_multi_videos.py \
    --data_dir /mnt/hanoverdev/data/patxiao/ECHO_numpy/20250126/ \
    --pretrained /mnt/hanoverdev/scratch/hanwen/xyliang/DINOv2_clip.pt \
    --embedding_dir /data/ECHO/dinov2_clip_slice_embeddings_multi_videos/ \
    --combine_embeddings \
    --image_size 256 \
    --batch_size 12 \
    --device cuda:3