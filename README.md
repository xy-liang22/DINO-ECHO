# ECHO
```bash
cd ECHO
conda create -n echo python=3.9
conda activate echo
pip install -r requirements.txt
```

## Dataset Processing
Dataset should be saved in a `.csv` file with structure:
```
label,split,<field-name>
1,train,<path-to-video-under-data-dictory>
0,val,<path-to-video-under-data-dictory>
0,test,<path-to-video-under-data-dictory>
...
```

## Classifying Tasks
### Finetune
```bash
# example in scripts/finetune.sh

python run.py --model dinov2_large_classifier \
                        --data_path <data-directory> \
                        --data_path_field <field-name> \
                        --dataset_csv <path-to-csv-file> \
                        --dataclass EchoData \
                        --output_dir <output-directory> \
                        --wandb_project ECHO \
                        --run_name <task>_v4 \
                        --batch_size 4 \
                        --image_size 256 \
                        --num_frames 64 \
                        --t_patch_size 8 \
                        --epochs 5 \
                        --warmup_epochs 1 \
                        --max_frames 128 \
                        --num_workers 20 \
                        --num_classes <number-of-classes> \
                        --blr 1e-3 \
                        --layer_decay 0.95 \
                        --weight_decay 0.05 \
                        --dropout 0.1 \
                        --smoothing 0.0 \
                        --fold 5 \
                        --pretrained <path-to-pretrained-DINOv2-model> \
                        --model_select val \
                        --balanced_dataset \
                        --device cuda:0

```
Checkpoint will be saved under `<output-directory>/fold_<fold-index>/`.

Best model will be saved in `<output-directory>/fold_<fold-index>/model_best.pth`.

Find log in `<output-directory>/fold_<fold-index>/log.jsonl`.

Find val result for each epoch in `<output-directory>/fold_<fold-index>/log_epoch.jsonl`.

Test of the best model will be conducted after finetuning finished, and the test result will be saved in `<output-directory>/fold_<fold-index>/test_results.json`

### Evaluate
```bash
# example in scripts/evaluate.sh

python run.py --model dinov2_large_classifier \
                        --data_path <data-directory> \
                        --data_path_field path \
                        --dataset_csv <path-to-csv-file> \
                        --dataclass EchoData \
                        --eval \
                        --eval_path <result-path> \
                        --batch_size 4 \
                        --image_size 256 \
                        --num_frames 64 \
                        --t_patch_size 8 \
                        --max_frames 128 \
                        --num_workers 20 \
                        --num_classes <number-of-classes> \
                        --smoothing 0.0 \
                        --fold 1 \
                        --resume <path-to-checkpoint> \
                        --device cuda:0
```

Evaluation result will be saved in `<result-path>`