import argparse

# ❗❗❗ TODO❗❗❗
def get_args_parser():
    parser = argparse.ArgumentParser(
        "MAE fine-tuning for image classification", add_help=False
    )
    parser.add_argument(
        "--batch_size",
        default=64,
        type=int,
        help="Batch size per GPU (effective batch size is batch_size * accum_iter * # gpus",
    )
    parser.add_argument("--epochs", default=50, type=int)
    parser.add_argument(
        "--accum_iter",
        default=1,
        type=int,
        help="Accumulate gradient iterations (for increasing the effective batch size under memory constraints)",
    )
    parser.add_argument(
        "--print_freq",
        default=20,
        type=int,
        help="Print frequency (default: 20)",
    )
    parser.add_argument(
        "--val_time",
        default=1,
        type=int,
        help="Time to run validation in an epoch",
    )

    # Model parameters
    parser.add_argument(
        "--model",
        default="vit_large_patch16",
        type=str,
        metavar="MODEL",
        help="Name of model to train",
    )
    
    parser.add_argument(
        "--clip_model_name",
        default="DINOv2_BiomedBERT_study_new",
        type=str,
        metavar="MODEL",
        help="Name of the CLIP model to use for classification",
    )
    
    parser.add_argument(
        "--num_transformer_layers",
        default=1,
        type=int,
        help="Number of transformer layers in transformer classifier layer",
    )
    
    parser.add_argument(
        "--num_heads",
        default=8,
        type=int,
        help="Number of attention heads in transformer classifier layer",
    )

    parser.add_argument("--input_size", default=224, type=int, help="images input size")

    parser.add_argument(
        "--dropout",
        type=float,
        default=0.5,
    )

    parser.add_argument(
        "--drop_path_rate",
        type=float,
        default=0.1,
        metavar="PCT",
        help="Drop path rate (default: 0.1)",
    )

    # Optimizer parameters
    parser.add_argument(
        "--clip_grad",
        type=float,
        default=None,
        metavar="NORM",
        help="Clip gradient norm (default: None, no clipping)",
    )
    parser.add_argument(
        "--weight_decay", type=float, default=0.05, help="weight decay (default: 0.05)"
    )
    
    parser.add_argument(
        "--hidden_dim",
        type=int,
        default=1024,
        metavar="N",
        help="hidden dimension of the classifier layer (default: 1024)",
    )

    parser.add_argument(
        "--lr",
        type=float,
        default=None,
        metavar="LR",
        help="learning rate (absolute lr)",
    )
    parser.add_argument(
        "--blr",
        type=float,
        default=1e-3,
        metavar="LR",
        help="base learning rate: absolute_lr = base_lr * total_batch_size / 256",
    )
    parser.add_argument(
        "--layer_decay",
        type=float,
        default=0.75,
        help="layer-wise lr decay from ELECTRA/BEiT",
    )

    parser.add_argument(
        "--min_lr",
        type=float,
        default=1e-6,
        metavar="LR",
        help="lower lr bound for cyclic schedulers that hit 0",
    )

    parser.add_argument(
        "--warmup_epochs", type=float, default=5, metavar="N", help="epochs to warmup LR"
    )

    # Augmentation parameters
    parser.add_argument(
        "--color_jitter",
        type=float,
        default=None,
        metavar="PCT",
        help="Color jitter factor (enabled only when not using Auto/RandAug)",
    )
    parser.add_argument(
        "--aa",
        type=str,
        default="rand-m7-mstd0.5-inc1",
        metavar="NAME",
        help='Use AutoAugment policy. "v0" or "original". " + "(default: rand-m9-mstd0.5-inc1)',
    ),
    parser.add_argument(
        "--smoothing", type=float, default=0.1, help="Label smoothing (default: 0.1)"
    )    
    parser.add_argument(
        "--RandFlipd_prob", type=float, default=0, help='RandFlipd probability'
    )
    parser.add_argument(
        "--RandRotate90d_prob", type=float, default=0, help='RandRotate90d probability'
    )
    # * Random Erase params
    parser.add_argument(
        "--reprob",
        type=float,
        default=0.25,
        metavar="PCT",
        help="Random erase prob (default: 0.25)",
    )
    parser.add_argument(
        "--remode",
        type=str,
        default="pixel",
        help='Random erase mode (default: "pixel")',
    )
    parser.add_argument(
        "--recount", type=int, default=1, help="Random erase count (default: 1)"
    )
    parser.add_argument(
        "--resplit",
        action="store_true",
        default=False,
        help="Do not random erase first (clean) augmentation split",
    )

    # * Mixup params
    parser.add_argument(
        "--mixup", type=float, default=0, help="mixup alpha, mixup enabled if > 0."
    )
    parser.add_argument(
        "--cutmix", type=float, default=0, help="cutmix alpha, cutmix enabled if > 0."
    )
    parser.add_argument(
        "--cutmix_minmax",
        type=float,
        nargs="+",
        default=None,
        help="cutmix min/max ratio, overrides alpha and enables cutmix if set (default: None)",
    )
    parser.add_argument(
        "--mixup_prob",
        type=float,
        default=1.0,
        help="Probability of performing mixup or cutmix when either/both is enabled",
    )
    parser.add_argument(
        "--mixup_switch_prob",
        type=float,
        default=0.5,
        help="Probability of switching to cutmix when both mixup and cutmix enabled",
    )
    parser.add_argument(
        "--mixup_mode",
        type=str,
        default="batch",
        help='How to apply mixup/cutmix params. Per "batch", "pair", or "elem"',
    )

    # * Finetuning params
    parser.add_argument("--pretrained", default="", help="The pretrained checkpoint")
    parser.add_argument("--freeze", action="store_true", help="freeze all layers", default=False)
    parser.add_argument("--global_pool", action="store_true")
    parser.add_argument('--feat_layer',     type=str, default='11', help='the layers to extract features that are used for downstream tasks')
    parser.add_argument("--init_ckpt", default="", help="Initialize from non-flash-attn checkpoint")
    parser.set_defaults(global_pool=True)
    parser.add_argument(
        "--cls_token",
        action="store_false",
        dest="global_pool",
        help="Use class token instead of global pool for classification",
    )
    parser.add_argument(
        "--num_classes",
        default=400,
        type=int,
        help="number of the classification types",
    )
    parser.add_argument(
        "--data_path",
        default="",
        help="the path of the dataset",
    )
    parser.add_argument(
        "--data_path_field",
        default="path",
        help="the field of data path in the dataset.csv file"
    )
    parser.add_argument(
        "--dataset_csv",
        default="",
        help="The csv file indicating the samples and labels",
    )
    parser.add_argument(
        "--prompt_path",
        default="",
        help="The path to the prompt file, which is a json file containing the prompts for each text and class",
    )
    parser.add_argument(
        "--task_name",
        default="LHF",
        type=str,
        help="The name of the task, for finding corresponding prompts",
    )
    parser.add_argument(
        "--output_dir",
        default="",
        help="path where to save, empty for no saving",
    )
    parser.add_argument(
        "--eval_path",
        default="",
        help="path to save evaluation result, empty for no saving",
    )
    parser.add_argument(
        "--log_dir",
        default="",
        help="path where to tensorboard log",
    )
    parser.add_argument(
        "--log_writer_type",
        type=str,
        default="wandb",
        choices=["wandb", "tensorboard"],
        help="type of log writer, wandb or tensorboard",
    )
    parser.add_argument(
        "--wandb_project",
        default="",
        help="wandb project name",
    )
    parser.add_argument(
        "--wandb_group",
        default="",
        help="wandb group name",
    )
    parser.add_argument(
        "--run_name",
        default="",
        help="wandb run name",
    )
    parser.add_argument(
        "--wandb_dir",
        default="./wandb",
        help="wandb dir",
    )
    parser.add_argument('--dataclass', type=str, default='EchoData', help='data reader') # UWCTData
    parser.add_argument(
        "--device", default="cuda:0", help="device to use for training / testing"
    )
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--resume", default="", help="resume from checkpoint")
    parser.add_argument("--save_freq", default=1, type=int, help="save frequency")
    parser.add_argument(
        "--start_epoch", default=0, type=int, metavar="N", help="start epoch"
    )
    parser.add_argument(
        "--fold", default=0, type=int, metavar="N", help="The fold index"
    )
    parser.add_argument(
        '--model_select', 
        type=str, 
        default='last_epoch', 
        help='model selection', 
        choices=['val', 'last_epoch', 'auroc']
    )
    parser.add_argument(
        '--N_val', type=int, default=5, help='validation frequency'
    )
    parser.add_argument(
        '--balanced_dataset',
        action='store_true',
        default=False,
        help='Weighted sampling'
    )
    parser.add_argument("--eval", action="store_true", help="Perform evaluation only")
    parser.add_argument("--predict", action="store_true", help="Perform prediction only")
    parser.add_argument("--n_bootstrap_eval", default=0, type=int, help="Number of bootstrap evaluations")
    parser.add_argument("--choose_mini", action="store_true", help="Choose a smaller training set for fast debugging")
    parser.add_argument(
        "--dist_eval",
        action="store_true",
        default=False,
        help="Enabling distributed evaluation (recommended during training for faster monitor",
    )
    parser.add_argument("--num_workers", default=10, type=int)
    parser.add_argument(
        "--pin_mem",
        action="store_true",
        help="Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.",
    )
    parser.add_argument("--no_pin_mem", action="store_false", dest="pin_mem")
    parser.set_defaults(pin_mem=True)

    # distributed training parameters
    parser.add_argument(
        "--world_size", default=1, type=int, help="number of distributed processes"
    )
    parser.add_argument("--local_rank", default=-1, type=int)
    parser.add_argument("--dist_on_itp", action="store_true")
    parser.add_argument(
        "--dist_url", default="env://", help="url used to set up distributed training"
    )

    # Video related configs
    parser.add_argument("--no_env", action="store_true")
    parser.add_argument("--rand_aug", default=False, action="store_true")
    parser.add_argument("--in_chans", default=1, type=int, help="images input size")
    parser.add_argument("--t_patch_size", default=2, type=int)
    parser.add_argument("--image_size", default=256, type=int)
    parser.add_argument("--num_frames", default=32, type=int)
    parser.add_argument("--max_frames", default=128, type=int)
    parser.add_argument("--view", default="coronal", type=str)
    parser.add_argument("--checkpoint_period", default=1, type=int)
    parser.add_argument("--sampling_rate", default=2, type=int)
    parser.add_argument("--distributed", action="store_true")
    parser.add_argument("--normalize_vol", action="store_true", default=False)
    parser.add_argument("--repeat_aug", default=1, type=int)
    parser.add_argument("--cpu_mix", action="store_true")
    parser.add_argument("--no_qkv_bias", action="store_true")
    parser.add_argument("--sep_pos_embed", action="store_true")
    parser.set_defaults(sep_pos_embed=True)
    parser.add_argument(
        "--fp16",
        action="store_true",
    )
    parser.set_defaults(fp16=True)
    parser.add_argument(
        "--jitter_scales_relative",
        default=[0.08, 1.0],
        type=float,
        nargs="+",
    )
    parser.add_argument(
        "--jitter_aspect_relative",
        default=[0.75, 1.3333],
        type=float,
        nargs="+",
    )
    parser.add_argument(
        "--threshold",
        default=0.5,
        type=float,
    )
    parser.add_argument("--cls_embed", action="store_true")
    parser.set_defaults(cls_embed=True)
    parser.add_argument("--use_ddp", action="store_true")
    parser.set_defaults(use_ddp=False)
    return parser