# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# DeiT: https://github.com/facebookresearch/deit
# BEiT: https://github.com/microsoft/unilm/tree/master/beit
# --------------------------------------------------------

import datetime
import json
import os

import sys
OPEN_CLIP_SRC_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'open_clip', 'src')
sys.path.insert(0, OPEN_CLIP_SRC_PATH)
print(f"Adding {OPEN_CLIP_SRC_PATH} to sys.path")

import time

import models

import custom_util.eval_dataset as eval_dataset
import custom_util.lr_decay as lrd
import custom_util.misc as misc

import numpy as np
import pandas as pd
import torch
import torch.backends.cudnn as cudnn
from pathlib import Path
from iopath.common.file_io import g_pathmgr as pathmgr
from run_engine import evaluate, run_one_epoch

from custom_util.decoder.mixup import MixUp as MixVideo
from custom_util.logging import master_print as print
from custom_util.eval_utils import create_transforms, get_split_func
from custom_util.misc import NativeScalerWithGradNormCount as NativeScaler
from custom_util.args_parser import get_args_parser
from custom_util.log_writing import CustomLogger

# from pytorchvideo.transforms.mix import MixVideo
from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy




def main(args):

    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
        if (os.path.exists(os.path.join(args.output_dir, "results.csv")) and not args.eval) or (args.eval and (os.path.exists(args.eval_path) or os.path.exists(os.path.join(args.output_dir, "results_bootstrap.csv")))):
            print("Output dir {} exists, skipping.".format(args.output_dir))
            return
    else:
        print("No output dir specified. Checkpoints and logs will not be saved.")
    
    if args.eval and args.eval_path and os.path.exists(args.eval_path):
        print(f"Eval path {args.eval_path} exists, skipping.")
        return

    if args.use_ddp:
        misc.init_distributed_mode(args)
    else:
        args.distributed = False

    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed + misc.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)

    cudnn.benchmark = True

    results = {}

    for fold_i in range(args.fold):
        _, test_stats_i = run_one_fold(args, device, fold_i)
        for k, v in test_stats_i.items():
            if k not in results:
                results[k] = []
            results[k].append(v)
    for k in results.keys():
        mean_v, std_v = np.mean(results[k]), np.std(results[k])
        print(f"Mean {k}: {mean_v:.4f} +/- {std_v:.4f}")
    # save the results to csv
    if args.output_dir:
        results = pd.DataFrame(results)
        results.to_csv(os.path.join(args.output_dir, "results.csv"), index=False)


def run_one_fold(args, device, fold_i):
    # set up finetuning dataset
    if args.dataclass in eval_dataset.__dict__:
        datacls = eval_dataset.__dict__[args.dataclass]
    else:
        raise ValueError(f"Unknown dataclass {args.dataclass}")
    split_func = get_split_func(args.dataclass) # set up the split function
    data_df = pd.read_csv(args.dataset_csv)
    train_splits, val_splits, test_splits = split_func(data_df, fold=fold_i, args=args)
    train_transform, val_transform = create_transforms(**vars(args))

    # whether make the dataset balanced
    if args.balanced_dataset:
        train_splits = eval_dataset.balance_dataset(train_splits[0], train_splits[1])
        # val_splits = eval_dataset.balance_dataset(val_splits[0], val_splits[1])

    dataset_train = datacls(df=data_df, splits=train_splits, processor=train_transform, **vars(args))
    dataset_val = datacls(df=data_df, splits=val_splits, processor=val_transform, **vars(args))
    dataset_test = datacls(df=data_df, splits=test_splits, processor=val_transform, **vars(args))
    # for testing
    #debug = dataset_train[150]

    if args.distributed:
        num_tasks = misc.get_world_size()
        global_rank = misc.get_rank()
        sampler_train = torch.utils.data.DistributedSampler(
            dataset_train, num_replicas=num_tasks, rank=global_rank, shuffle=True
        )
        # print("Sampler_train = %s" % str(sampler_train))
        if args.dist_eval:
            if len(dataset_val) % num_tasks != 0:
                print(
                    "Warning: Enabling distributed evaluation with an eval dataset not divisible by process number. "
                    "This will slightly alter validation results as extra duplicate entries are added to achieve "
                    "equal num of samples per-process."
                )
            sampler_val = torch.utils.data.DistributedSampler(
                dataset_val, num_replicas=num_tasks, rank=global_rank, shuffle=True
            ) # shuffle=True to reduce monitor bias
        else:
            sampler_val = torch.utils.data.SequentialSampler(dataset_val)
    else:
        num_tasks = 1
        global_rank = 0
        sampler_train = torch.utils.data.RandomSampler(dataset_train)
        sampler_val = torch.utils.data.SequentialSampler(dataset_val)

    data_loader_train = torch.utils.data.DataLoader(
        dataset_train,
        sampler=sampler_train,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        # drop_last=True,
    )
    data_loader_val = torch.utils.data.DataLoader(
        dataset_val,
        sampler=sampler_val,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=False,
    )
    data_loader_test = torch.utils.data.DataLoader(
        dataset_test,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=False,
    )

    # ❗❗❗ TODO❗❗❗
    mixup_fn = None
    mixup_active = args.mixup > 0 or args.cutmix > 0.0 or args.cutmix_minmax is not None
    if mixup_active:
        print("Mixup is activated!")
        mixup_fn = MixVideo(
            mixup_alpha=args.mixup,
            cutmix_alpha=args.cutmix,
            mix_prob=args.mixup_prob,
            switch_prob=args.mixup_switch_prob,
            label_smoothing=args.smoothing,
            num_classes=args.num_classes,
        )
    
    # define the model
    model = models.__dict__[args.model](**vars(args))
    model.to(device)
    
    if args.resume:
        msg = model.load_state_dict(torch.load(args.resume), strict=False)
        print(f"Loaded model from {args.resume} with msg {msg}")
    
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    # print("Model = %s" % str(model_without_ddp))
    print("number of params (M): %.4f" % (n_parameters / 1.0e6))
    
    if args.eval:
        test_stats = evaluate(data_loader_test, model, device, n_bootstrap_eval=args.n_bootstrap_eval)
        if args.eval_path:
            with pathmgr.open(args.eval_path, "w") as f:
                json.dump(test_stats, f, indent=4)
        if args.n_bootstrap_eval and args.output_dir:
            test_stats = pd.DataFrame(test_stats)
            test_stats.to_csv(os.path.join(args.output_dir, "results_bootstrap.csv"), index=False)
            print("Saved bootstrap evaluation results to %s" % os.path.join(args.output_dir, "results_bootstrap.csv"))
        exit(0)

    eff_batch_size = (
        args.batch_size * args.accum_iter * misc.get_world_size() * args.repeat_aug
    )
    
    if args.lr is None:  # only base_lr is specified
        args.lr = args.blr * eff_batch_size / 256

    print("base lr: %.2e" % (args.lr * 256 / eff_batch_size))
    print("actual lr: %.2e" % args.lr)
    print("accumulate grad iterations: %d" % args.accum_iter)
    print("effective batch size: %d" % eff_batch_size)

    model_without_ddp = model
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[torch.cuda.current_device()]
        )
        model_without_ddp = model.module

    def freeze_all_except_classifier():
        for n, p in model_without_ddp.named_parameters():
            if not n.startswith("classifier"):
                p.requires_grad = False
            else:
                p.requires_grad = True

    # ❗❗❗ TODO❗❗❗
    if args.freeze:
        # freeze blocks
        freeze_all_except_classifier()

    # ❗❗❗ TODO❗❗❗: whether cnn is used
    # build optimizer with layer-wise lr decay (lrd)
    if not 'cnn' in args.model:
        # vit class models
        param_groups = lrd.param_groups_lrd(
            model_without_ddp,
            args.weight_decay,
            no_weight_decay_list=model_without_ddp.no_weight_decay(),
            layer_decay=args.layer_decay,
        )
        optimizer = torch.optim.AdamW(param_groups, lr=args.lr)
    else:
        # cnn class models
        param_groups = model_without_ddp.parameters()
        optimizer = torch.optim.AdamW(param_groups, lr=args.lr, weight_decay=args.weight_decay)

    loss_scaler = NativeScaler()

    # ❗❗❗ TODO❗❗❗
    if mixup_fn is not None:
        # smoothing is handled with mixup label transform
        criterion = SoftTargetCrossEntropy()
    elif args.smoothing > 0.0 and dataset_train.mode != 'multilabel':
        criterion = LabelSmoothingCrossEntropy(smoothing=args.smoothing)
    elif dataset_train.mode == 'multilabel':
        criterion = torch.nn.BCEWithLogitsLoss()
    else:
        criterion = torch.nn.CrossEntropyLoss()

    print("criterion = %s" % str(criterion))

    if args.output_dir:
        # set up fold output dir
        args.fold_output_dir = os.path.join(args.output_dir, f"fold_{fold_i}")
        os.makedirs(args.fold_output_dir, exist_ok=True)
    
    log_writer = None
    # set up logging
    if global_rank == 0:
        args.fold_run_name = args.run_name + f"_fold_{fold_i}"
        log_writer = CustomLogger(args)

    print(f"Start training for {args.epochs} epochs")
    start_time = time.time()
    max_auroc, best_loss, best_model = 0.0, 1e5, None
    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            data_loader_train.sampler.set_epoch(epoch)
        train_stats, val_stats = run_one_epoch(
            model,
            criterion,
            data_loader_train,
            data_loader_val,
            optimizer,
            device,
            epoch,
            loss_scaler,
            args.clip_grad,
            mixup_fn,
            log_writer=log_writer,
            args=args,
            fp16=args.fp16,
        )
        if args.fold_output_dir:
            if (epoch + 1) % args.save_freq == 0 or epoch == args.epochs - 1:
                checkpoint_path = misc.save_model(
                    args=args,
                    model=model,
                    model_without_ddp=model_without_ddp,
                    optimizer=optimizer,
                    loss_scaler=loss_scaler,
                    epoch=epoch,
                )

        log_stats = {
            "epoch": epoch,
            **{
                f"train_{k}": float(f"{v:4f}") if "lr" not in k \
                else float(f"{v:5g}") for k, v in train_stats.items()
            },
            **{
                f"val_{k}": float(f"{v:4f}") for k, v in val_stats.items()
            },
            "n_parameters": n_parameters,
        }

        # save val results to log
        if args.fold_output_dir and misc.is_main_process():
            log_writer.flush(log_stats, f"{args.fold_output_dir}/log_epoch.jsonl")

        # save the best model
        if args.model_select == "val":
            if val_stats["loss"] < best_loss:
                best_loss = val_stats["loss"]
                best_model = model.state_dict()
        elif args.model_select == "auroc":
            if val_stats["AUROC"] > max_auroc:
                max_auroc = val_stats["AUROC"]
                best_model = model.state_dict()
        elif args.model_select == "last_epoch":
            if epoch == args.epochs - 1:
                best_model = model.state_dict()
        else:
            raise ValueError(f"Unknown model selection information {args.model_select}")
    
    # test
    model.load_state_dict(best_model)
    print("Testing the best model")
    test_stats = evaluate(data_loader_test, model, device)
    
    print(f"Max AUROC: {max_auroc:.4f}")
    print(f"BACC: {test_stats['BACC']:.4f}, F1: {test_stats['F1 Score']:.4f}, ACC: {test_stats['Accuracy']:.4f}, AUROC: {test_stats['AUROC']:.4f}, AUPRC: {test_stats['AUPRC']:.4f}, Precision: {test_stats['Precision']:.4f}, Recall: {test_stats['Recall']:.4f}")
    print(f"Test loss: {test_stats['loss']:.4f}")
    
    log_writer.update({
        "test/bacc": test_stats["BACC"],
        "test/acc": test_stats["Accuracy"],
        "test/auroc": test_stats["AUROC"],
        "test/precision": test_stats["Precision"],
        "test/recall": test_stats["Recall"],
        "test/auprc": test_stats["AUPRC"],
        "test/f1_score": test_stats["F1 Score"],
    }, step=(epoch + 1) * 1000 * args.repeat_aug)
    log_writer.close()
    
    
    log_stats = {
        **{f"test_{k}": float(f"{v:.4f}") for k, v in test_stats.items()},
        "n_parameters": n_parameters,
    }
    
    
    if args.fold_output_dir and misc.is_main_process():
        # save test results to log
        with pathmgr.open(f"{args.fold_output_dir}/log.jsonl", "a") as f:
            f.write(json.dumps(log_stats) + "\n")
        with pathmgr.open(f"{args.fold_output_dir}/test_results.json","w",) as f:
            json.dump(test_stats, f, indent=4)
            
        # save the best model
        checkpoint_path = os.path.join(args.fold_output_dir, "model_best.pth")
        torch.save(best_model, checkpoint_path)

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print("Training time {}".format(total_time_str))
    return [checkpoint_path], test_stats


if __name__ == "__main__":
    global args
    args = get_args_parser()
    args = args.parse_args()
    main(args)

    """
    screen -S heart_failure_test
    CTRL+A and D
    screen -r heart_failure_test
    """