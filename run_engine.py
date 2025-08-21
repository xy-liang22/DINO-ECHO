# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# DeiT: https://github.com/facebookresearch/deit
# BEiT: https://github.com/microsoft/unilm/tree/master/beit
# --------------------------------------------------------

import math
import sys
from typing import Iterable, Optional

import custom_util.lr_sched as lr_sched
import custom_util.misc as misc
from custom_util.eval_metrics import get_classification_metrics
import torch

from custom_util.logging import master_print as print
from timm.data import Mixup

import json
from iopath.common.file_io import g_pathmgr as pathmgr

import numpy as np

def run_one_epoch(
    model: torch.nn.Module,
    criterion: torch.nn.Module,
    data_loader: Iterable,
    data_loader_val: Iterable,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    loss_scaler,
    max_norm: float = 0,
    mixup_fn: Optional[Mixup] = None,
    log_writer=None,
    args=None,
    fp16=False,
):
    model.train(True)
    metric_logger = misc.MetricLogger(delimiter="  ")
    metric_logger.add_meter("lr", misc.SmoothedValue(window_size=1, fmt="{value:.6f}"))
    metric_logger.add_meter(
        "cpu_mem", misc.SmoothedValue(window_size=1, fmt="{value:.6f}")
    )
    metric_logger.add_meter(
        "cpu_mem_all", misc.SmoothedValue(window_size=1, fmt="{value:.6f}")
    )
    metric_logger.add_meter(
        "gpu_mem", misc.SmoothedValue(window_size=1, fmt="{value:.6f}")
    )
    header = "Epoch: [{}]".format(epoch)

    accum_iter = args.accum_iter
    val_iter_float = len(data_loader) / (args.val_time + 1)
    cnt_val_time = 0

    optimizer.zero_grad()


    for data_iter_step, (samples, targets) in enumerate(
        metric_logger.log_every(data_loader, args.print_freq, header)
    ):

        # we use a per iteration (instead of per epoch) lr scheduler
        if data_iter_step % accum_iter == 0:
            lr_sched.adjust_learning_rate(
                optimizer, data_iter_step / len(data_loader) + epoch, args
            )

        # ❗❗❗ TODO❗❗❗
        if args.cpu_mix:
            if mixup_fn is not None:
                samples, targets = mixup_fn(samples, targets)
            if isinstance(samples, (list, tuple)):
                samples = [s.to(device, non_blocking=True) for s in samples]
            else:
                samples = samples.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
        else:
            if isinstance(samples, (list, tuple)):
                samples = [s.to(device, non_blocking=True) for s in samples]
            else:
                samples = samples.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            if mixup_fn is not None:
                samples, targets = mixup_fn(samples, targets)

        with torch.cuda.amp.autocast(dtype=torch.float16 if fp16 else None):
            outputs = model(samples)
            loss = criterion(outputs, targets)

        loss_value = loss.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)

        loss /= accum_iter
        loss_scaler(
            loss,
            optimizer,
            clip_grad=max_norm,
            parameters=model.parameters(),
            create_graph=False,
            update_grad=(data_iter_step + 1) % accum_iter == 0,
        )
        if (data_iter_step + 1) % accum_iter == 0:
            optimizer.zero_grad()

        torch.cuda.synchronize()

        metric_logger.update(loss=loss_value)
        metric_logger.update(cpu_mem=misc.cpu_mem_usage()[0])
        metric_logger.update(cpu_mem_all=misc.cpu_mem_usage()[1])
        metric_logger.update(gpu_mem=misc.gpu_mem_usage())
        min_lr = 10.0
        max_lr = 0.0
        for group in optimizer.param_groups:
            min_lr = min(min_lr, group["lr"])
            max_lr = max(max_lr, group["lr"])

        metric_logger.update(lr=max_lr)

        loss_value_reduce = misc.all_reduce_mean(loss_value)
        
        # Update training curve
        if(data_iter_step + 1) % accum_iter == 0:
            """We use epoch_1000x as the x-axis in tensorboard.
            This calibrates different curves when batch size changes.
            """
            epoch_1000x = int(
                (data_iter_step / len(data_loader) + epoch) * 1000 * args.repeat_aug
            )
            log_writer.update({
                "train/loss": loss_value_reduce,
                "train/lr": max_lr,
                "train/iter_ratio": data_iter_step / len(data_loader),
            }, epoch_1000x)
        
        # Check if it is time to validate
        if data_iter_step < len(data_loader) * (cnt_val_time + 1) / args.val_time and \
            data_iter_step + 1 >= len(data_loader) * (cnt_val_time + 1) / args.val_time:
            
            val_stats = evaluate(data_loader_val, model, device)
            log_stats = {
                "epoch": epoch,
                "val_time": cnt_val_time,
                "train_lr": float(f"{metric_logger.lr.global_avg:.5g}"),
                "train_loss": float(f"{metric_logger.loss.global_avg:.4f}"),
                **{f"val_{k}": float(f"{v:.4f}") for k, v in val_stats.items()},
            }
            # Output check result
            if args.fold_output_dir and misc.is_main_process():
                with pathmgr.open(f"{args.fold_output_dir}/log.jsonl", "a") as f:
                    f.write(json.dumps(log_stats) + "\n")
            # Update validation result with log_writer
            epoch_1000x = int(
                (data_iter_step / len(data_loader) + epoch) * 1000 * args.repeat_aug
            )
            val_update = {
                "val/loss": val_stats["loss"],
                "val/bacc": val_stats["BACC"],
                "val/acc": val_stats["Accuracy"],
                "val/auroc": val_stats["AUROC"],
                "val/precision": val_stats["Precision"],
                "val/recall": val_stats["Recall"],
                "val/auprc": val_stats["AUPRC"],
                "val/f1_score": val_stats["F1 Score"],
            }
            if data_loader.dataset.mode == "multiclass":
                val_update.update({
                    "val/micro_auroc": val_stats["micro AUROC"],
                    "val/weighted_auroc": val_stats["weighted AUROC"],
                    "val/micro_auprc": val_stats["micro AUPRC"],
                    "val/micro_recall": val_stats["micro Recall"],
                    "val/cohen_kappa": val_stats["Cohen's Kappa"],
                })
                for label in data_loader.dataset.label_dict.keys():
                    val_update[f"val/auroc_{label}"] = val_stats[f"AUROC ({label})"]
                    val_update[f"val/auprc_{label}"] = val_stats[f"AUPRC ({label})"]
                    val_update[f"val/recall_{label}"] = val_stats[f"Recall ({label})"]
            log_writer.update(val_update, epoch_1000x)
            cnt_val_time += 1
            model.train(True)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}, val_stats


@torch.no_grad()
def evaluate(data_loader, model, device, return_pred=False, n_bootstrap_eval=0):
    # ❗❗❗ TODO❗❗❗: whether use mixup in evaluation
    criterion = torch.nn.CrossEntropyLoss() if data_loader.dataset.mode != "multilabel" else torch.nn.BCEWithLogitsLoss()

    metric_logger = misc.MetricLogger(delimiter="  ")
    header = "Test:"

    # switch to evaluation mode
    model.eval()
    all_preds, all_targets = [], []
    for images, target in metric_logger.log_every(data_loader, 10, header):
        if isinstance(images, (list, tuple)):
            images = [img.to(device, non_blocking=True) for img in images]
        else:
            images = images.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)
        
        # compute output
        with torch.cuda.amp.autocast():
            output = model(images)
            loss = criterion(output, target)
        all_preds.append(output)
        all_targets.append(target)

        if isinstance(images, (list, tuple)):
            batch_size = len(images[0])
        else:
            batch_size = images.shape[0]
        metric_logger.update(loss=loss.item())
    
    output, target = torch.cat(all_preds), torch.cat(all_targets)
    if n_bootstrap_eval:
        return bootstarp_metrics(output, target, data_loader, n_bootstrap=n_bootstrap_eval)
    
    eval_stats = get_classification_metrics(output, target, data_loader.dataset.label_dict, mode=data_loader.dataset.mode)
    metric_logger.meters["bacc"].update(eval_stats["BACC"], n=batch_size)
    metric_logger.meters["acc"].update(eval_stats["Accuracy"], n=batch_size)
    metric_logger.meters["f1_score"].update(eval_stats["F1 Score"], n=batch_size)
    metric_logger.meters["auroc"].update(eval_stats["AUROC"], n=batch_size)
    metric_logger.meters["precision"].update(eval_stats["Precision"], n=batch_size)
    metric_logger.meters["recall"].update(eval_stats["Recall"], n=batch_size)
    metric_logger.meters["auprc"].update(eval_stats["AUPRC"], n=batch_size)
    
    if data_loader.dataset.mode == "binary":
        # gather the stats from all processes
        metric_logger.synchronize_between_processes()
        print(
            "* Accuracy {top1.global_avg:.4f} BACC {top3.global_avg:.4f} Precision {top5.global_avg:.4f} Recall {top7.global_avg:.4f} F1 Score {f1.global_avg:.4f} AUROC {auroc.global_avg:.4f} AUPRC {auprc.global_avg:.4f} loss {losses.global_avg:.4f}".format(
                top1=metric_logger.acc,
                top3=metric_logger.bacc,
                top5=metric_logger.precision,
                top7=metric_logger.recall,
                f1=metric_logger.f1_score,
                auroc=metric_logger.auroc,
                auprc=metric_logger.auprc,
                losses=metric_logger.loss
            )
        )
    elif data_loader.dataset.mode == "multiclass":
        metric_logger.meters["micro_auroc"].update(eval_stats["micro AUROC"], n=batch_size)
        metric_logger.meters["weighted_auroc"].update(eval_stats["weighted AUROC"], n=batch_size)
        metric_logger.meters["micro_auprc"].update(eval_stats["micro AUPRC"], n=batch_size)
        metric_logger.meters["micro_recall"].update(eval_stats["micro Recall"], n=batch_size)
        metric_logger.meters["cohen_kappa"].update(eval_stats["Cohen's Kappa"], n=batch_size)
        for label in data_loader.dataset.label_dict.keys():
            metric_logger.meters[f"auroc_{label}"].update(eval_stats[f"AUROC ({label})"], n=batch_size)
            metric_logger.meters[f"auprc_{label}"].update(eval_stats[f"AUPRC ({label})"], n=batch_size)
            metric_logger.meters[f"recall_{label}"].update(eval_stats[f"Recall ({label})"], n=batch_size)
        metric_logger.synchronize_between_processes()
        
        print(
            "* Accuracy {top1.global_avg:.4f} BACC {top3.global_avg:.4f} Precision {top5.global_avg:.4f} Recall {top7.global_avg:.4f} micro_Recall {micro_recall.global_avg:.4f} F1 Score {f1.global_avg:.4f} AUROC {auroc.global_avg:.4f} micro_AUROC {micro_auroc.global_avg:.4f} weighted_AUROC {weighted_auroc.global_avg:.4f} AUPRC {auprc.global_avg:.4f} micro_AUPRC {micro_auprc.global_avg:.4f} loss {losses.global_avg:.4f} kappa {kappa.global_avg:.4f}".format(
                top1=metric_logger.acc,
                top3=metric_logger.bacc,
                top5=metric_logger.precision,
                top7=metric_logger.recall,
                micro_recall=metric_logger.micro_recall,
                f1=metric_logger.f1_score,
                auroc=metric_logger.auroc,
                micro_auroc=metric_logger.micro_auroc,
                weighted_auroc=metric_logger.weighted_auroc,
                auprc=metric_logger.auprc,
                micro_auprc=metric_logger.micro_auprc,
                losses=metric_logger.loss,
                kappa=metric_logger.cohen_kappa
            )
        )
        for label in data_loader.dataset.label_dict.keys():
            print(
                f"* AUROC ({label}) {metric_logger.meters[f'auroc_{label}'].global_avg:.4f} AUPRC ({label}) {metric_logger.meters[f'auprc_{label}'].global_avg:.4f} Recall ({label}) {metric_logger.meters[f'recall_{label}'].global_avg:.4f}"
            )
    eval_stats["loss"] = metric_logger.loss.global_avg

    return eval_stats


def bootstarp_metrics(output, target, data_loader, n_bootstrap=1000):
    print(f"Bootstrapping metrics with {n_bootstrap} iterations")
    bootstrap_results = {}
    idx = 0
    while True:
        indices = np.random.choice(len(target), len(target), replace=True)
        output_i = output[indices]
        target_i = target[indices]
        results = get_classification_metrics(output_i, target_i, data_loader.dataset.label_dict, mode=data_loader.dataset.mode)
        assert results is not None, "Results should not be None"
        for metric in results:
            if metric not in bootstrap_results:
                bootstrap_results[metric] = []
            bootstrap_results[metric].append(results[metric])
        idx += 1
        if idx % 100 == 0:
            print(f"Bootstrap iteration {idx}/{n_bootstrap} completed")
        if idx >= n_bootstrap:
            break
    return bootstrap_results