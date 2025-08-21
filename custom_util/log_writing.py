import wandb
from torch.utils.tensorboard import SummaryWriter
from iopath.common.file_io import g_pathmgr as pathmgr
import json

class CustomLogger:
    """
    Custom logger class to handle logging with different frameworks.
    """

    def __init__(self, args):
        self.log_writer_type = args.log_writer_type
        
        if args.log_writer_type == "wandb":
            wandb.init(
                project=args.wandb_project,
                group=args.wandb_group,
                name=args.fold_run_name,
                dir=args.wandb_dir,
                id=args.fold_run_name,
                config=vars(args),
                reinit=True
            )
            self.log_writer = wandb
        elif args.log_writer_type == "tensorboard" and args.log_dir is not None:
            try:
                pathmgr.mkdirs(args.log_dir)
            except Exception as _:
                pass
            self.log_writer = SummaryWriter(log_dir=args.log_dir)
        else:
            self.log_writer = None
            print("No log writer selected. Please select either wandb or tensorboard.")


    def update(self, loss_dict, step):
        """
        Update log writer with loss dict
        """
        if self.log_writer is not None:
            if self.log_writer_type == "wandb":
                for k, v in loss_dict.items():
                    self.log_writer.log({k: v}, step=step)
            elif self.log_writer_type == "tensorboard":
                for k, v in loss_dict.items():
                    self.log_writer.add_scalar(k, v, global_step=step)

    def flush(self, log_stats, output_path=None):
        """
        Flush log writer
        """
        if self.log_writer is not None:
            if self.log_writer_type == "tensorboard":
                self.log_writer.flush()
        with pathmgr.open(output_path, "a") as f:
            f.write(json.dumps(log_stats) + "\n")

    def close(self):
        """
        Close log writer
        """
        if self.log_writer is not None:
            if self.log_writer_type == "wandb":
                self.log_writer.finish()
            elif self.log_writer_type == "tensorboard":
                self.log_writer.flush()
                self.log_writer.close()