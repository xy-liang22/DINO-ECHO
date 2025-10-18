# python other/save_dinov2_embeddings_multi_videos.py \
#     --data_dir /mnt/hanoverdev/data/patxiao/ECHO_numpy/original_size/ \
#     --pretrained /mnt/hanoverdev/scratch/hanwen/xyliang/DINOv2_study_original1.pt \
#     --embedding_dir /data/ECHO/dinov2_study_original1_embeddings_multi_videos/ \
#     --combined_embedding_path /data/ECHO/dinov2_study_original1_embeddings_multi_videos_mean.pt \
#     --config_path models/dinov2_modules/configs/train/vitl16_lbsz96_500ep_resume_from_general.yaml \
#     --combine_embeddings \
#     --mean_embeddings \
#     --image_size 224 \
#     --batch_size 12 \
#     --device cuda:3


# python other/save_dinov2_embeddings.py \
#     --data_dir /mnt/hanoverdev/data/patxiao/ECHO_numpy/original_size/ \
#     --pretrained /mnt/hanoverdev/scratch/hanwen/xyliang/DINOv2_study_original1.pt \
#     --embedding_dir /data/ECHO/dinov2_study_original1_embeddings/ \
#     --combined_embedding_path /data/ECHO/dinov2_study_original1_embeddings_mean.pt \
#     --config_path models/dinov2_modules/configs/train/vitl16_lbsz96_500ep_resume_from_general.yaml \
#     --combine_embeddings \
#     --mean_embeddings \
#     --image_size 224 \
#     --batch_size 12 \
#     --device cuda:0


python other/save_dinov2_public_embeddings.py \
    --embedding_dir /data/ECHO/dinov2_public_embeddings/ \
    --combined_embedding_path /data/ECHO/dinov2_public_embeddings_mean.pt \
    --image_size 256 \
    --batch_size 12 \
    --mean_embeddings \
    --device cuda:3

# python other/save_dinov2_embeddings.py \
#     --data_dir /mnt/hanoverdev/data/patxiao/ECHO_numpy/original_size/ \
#     --pretrained /mnt/hanoverdev/scratch/hanwen/exp/echofound/pretrain_dinov2/20250529_vitl16_lbsz96_gbsz384_500ep_resume_general/eval/training_399999/teacher_checkpoint.pth \
#     --config_path models/dinov2_modules/configs/train/vitl16_lbsz96_500ep_resume_from_general.yaml \
#     --embedding_dir /data/ECHO/dinov2_original1_fullsize_embeddings/ \
#     --combined_embedding_path /data/ECHO/dinov2_original1_fullsize_embeddings_mean.pt \
#     --combine_embeddings \
#     --mean_embeddings \
#     --image_size 224 \
#     --batch_size 24 \
#     --device cuda:1

# python other/save_dinov2_transformer_embeddings.py \
#     --data_dir /mnt/hanoverdev/data/patxiao/ECHO_numpy/original_size/ \
#     --pretrained /mnt/hanoverdev/scratch/hanwen/xyliang/DINOv2_transformer_DINO_only.pt \
#     --ckpt_path /mnt/hanoverdev/scratch/hanwen/xyliang/DINOv2_transformer.pt \
#     --config_path /home/xuhw/xyliang/research-projects/ECHO/models/dinov2_modules/configs/train/vitl16_lbsz96_500ep_resume_from_general.yaml \
#     --embedding_dir /data/ECHO/dinov2_transformer_embeddings/ \
#     --combined_embedding_path /data/ECHO/dinov2_transformer_embeddings_mean.pt \
#     --combine_embeddings \
#     --mean_embeddings \
#     --image_size 256 \
#     --batch_size 12 \
#     --device cuda:0

# python other/save_biomedclip_embeddings.py \
#     --data_dir /mnt/hanoverdev/data/patxiao/ECHO_numpy/original_size/ \
#     --embedding_dir /data/ECHO/biomedclip_embeddings/ \
#     --combined_embedding_path /data/ECHO/biomedclip_embeddings_mean.pt \
#     --combine_embeddings \
#     --mean_embeddings \
#     --image_size 224 \
#     --batch_size 12 \
#     --device cuda:3