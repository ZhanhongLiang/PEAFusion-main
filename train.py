# Modified by Yan Wang based on the following repositories.
# CRM_RGBTSeg: https://github.com/UkcheolShin/CRM_RGBTSeg

import os
import os.path as osp
from argparse import ArgumentParser

from mmcv import Config
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import Callback, ModelCheckpoint, RichModelSummary, RichProgressBar
from pytorch_lightning.loggers import CSVLogger
from torch.utils.data import DataLoader, WeightedRandomSampler
import torch

from models import MODELS
from dataloaders import build_dataset

# MaskFormer
from detectron2.config import get_cfg
from detectron2.engine import default_setup
from detectron2.projects.deeplab import add_deeplab_config
from models.mask2former import add_maskformer2_config
from models.config import add_peafusion_config
from util.RGBTCheckpointer import RGBTCheckpointer

from copy import deepcopy

import pickle
import numpy as np

try:
    # 5678 is the default attach port in the VS Code debug configurations. Unless a host and port are specified, host defaults to 127.0.0.1
    debugpy.listen(("localhost", 9501))
    print("Waiting for debugger attach")
    debugpy.wait_for_client()
except Exception as e:
    pass


class EpochMetricsFileLogger(Callback):
    def __init__(self, log_path, step_log_path, experiment_name, config_file, cfg, step_log_interval=20):
        super().__init__()
        self.log_path = log_path
        self.step_log_path = step_log_path
        self.experiment_name = experiment_name
        self.config_file = config_file
        self.cfg = cfg
        self.step_log_interval = step_log_interval

    @staticmethod
    def _to_scalar(value):
        if isinstance(value, torch.Tensor):
            if value.numel() == 1:
                return value.detach().cpu().item()
            return None
        if isinstance(value, (int, float)):
            return value
        return None

    @staticmethod
    def _format_metric_value(value):
        if isinstance(value, float):
            return f"{value:.6f}"
        return str(value)

    def on_fit_start(self, trainer, pl_module):
        model = pl_module.network
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        state_dict_keys = len(model.state_dict())
        parameter_keys = len(list(model.named_parameters()))
        module_keys = len(list(model.named_modules()))

        with open(self.log_path, "w", encoding="utf-8") as f:
            f.write("=== Experiment Summary ===\n")
            f.write(f"name: {self.experiment_name}\n")
            f.write(f"config_file: {self.config_file}\n")
            f.write(f"dataset_name: {self.cfg.DATASETS.NAME}\n")
            f.write(f"dataset_dir: {self.cfg.DATASETS.DIR}\n")
            f.write(f"batch_size: {self.cfg.SOLVER.IMS_PER_BATCH}\n")
            f.write(f"use_semoe_fusion: {self.cfg.MODEL.FUSION.USE_SEMOE_FUSION}\n")
            f.write(f"fusion_type: {self.cfg.MODEL.FUSION.FUSION_TYPE}\n")
            f.write(f"router_type: {self.cfg.MODEL.FUSION.ROUTER_TYPE}\n")
            f.write(f"decoder_type: {self.cfg.MODEL.MASK_FORMER.DECODER_TYPE}\n")
            f.write(f"recursive_rerouting: {self.cfg.MODEL.MASK_FORMER.RECURSIVE_REROUTING}\n")
            f.write(f"class_embed_dim: {self.cfg.MODEL.FUSION.CLASS_EMBED_DIM}\n")
            f.write(f"expert_depth: {self.cfg.MODEL.FUSION.EXPERT_DEPTH}\n")
            f.write(f"loss_balance_weight: {self.cfg.MODEL.MASK_FORMER.LOSS_BALANCE_WEIGHT}\n")
            f.write(f"num_classes: {self.cfg.MODEL.SEM_SEG_HEAD.NUM_CLASSES}\n")
            f.write("\n=== Model Stats ===\n")
            f.write(f"meta_architecture: {self.cfg.MODEL.META_ARCHITECTURE}\n")
            f.write(f"model_class: {model.__class__.__name__}\n")
            f.write(f"backbone_class: {model.backbone.__class__.__name__}\n")
            f.write(f"sem_seg_head_class: {model.sem_seg_head.__class__.__name__}\n")
            f.write(f"predictor_class: {model.sem_seg_head.predictor.__class__.__name__}\n")
            f.write(f"total_params: {total_params}\n")
            f.write(f"trainable_params: {trainable_params}\n")
            f.write(f"non_trainable_params: {total_params - trainable_params}\n")
            f.write(f"state_dict_keys: {state_dict_keys}\n")
            f.write(f"named_parameter_keys: {parameter_keys}\n")
            f.write(f"named_module_keys: {module_keys}\n")
            f.write("\n=== Model Structure ===\n")
            f.write(repr(model))
            f.write("\n\n=== Epoch Metrics ===\n")
        with open(self.step_log_path, "w", encoding="utf-8") as f:
            f.write("=== Train Step Metrics ===\n")

    def _append_metrics(self, trainer, stage):
        metrics = trainer.callback_metrics
        payload = {
            "epoch": trainer.current_epoch,
            "global_step": trainer.global_step,
            "stage": stage,
        }
        for key in [
            "train/total_loss",
            "val/average_precision",
            "val/average_recall",
            "val/average_IoU",
            "val/allAcc",
        ]:
            if key in metrics:
                scalar = self._to_scalar(metrics[key])
                if scalar is not None:
                    payload[key] = scalar

        line_parts = [
            f"epoch: {payload['epoch']}",
            f"step: {payload['global_step']}",
            f"stage: {payload['stage']}",
        ]
        if "train/total_loss" in payload:
            line_parts.append(f"loss: {self._format_metric_value(payload['train/total_loss'])}")
        if "val/average_precision" in payload:
            line_parts.append(f"precision: {self._format_metric_value(payload['val/average_precision'])}")
        if "val/average_recall" in payload:
            line_parts.append(f"recall: {self._format_metric_value(payload['val/average_recall'])}")
        if "val/average_IoU" in payload:
            line_parts.append(f"mIoU: {self._format_metric_value(payload['val/average_IoU'])}")
        if "val/allAcc" in payload:
            line_parts.append(f"allAcc: {self._format_metric_value(payload['val/allAcc'])}")
        if stage == "val_epoch_end":
            checkpoint_callback = getattr(trainer, "checkpoint_callback", None)
            if checkpoint_callback is not None:
                best_score = checkpoint_callback.best_model_score
                if best_score is not None:
                    best_score = self._to_scalar(best_score)
                    if best_score is not None:
                        line_parts.append(f"best_mIoU: {self._format_metric_value(best_score)}")
                best_path = checkpoint_callback.best_model_path
                if best_path:
                    line_parts.append(f"best_ckpt: {best_path}")

        with open(self.log_path, "a", encoding="utf-8") as f:
            f.write("    ".join(line_parts) + "\n")

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if trainer.global_step == 0 or trainer.global_step % self.step_log_interval != 0:
            return
        scalar_loss = self._to_scalar(trainer.callback_metrics.get("train/total_loss"))
        if scalar_loss is None:
            return
        with open(self.step_log_path, "a", encoding="utf-8") as f:
            f.write(
                f"epoch: {trainer.current_epoch}    "
                f"step: {trainer.global_step}    "
                f"loss: {self._format_metric_value(scalar_loss)}\n"
            )

    def on_train_epoch_end(self, trainer, pl_module):
        self._append_metrics(trainer, "train_epoch_end")

    def on_validation_end(self, trainer, pl_module):
        stage = "sanity_check_end" if trainer.sanity_checking else "val_epoch_end"
        self._append_metrics(trainer, stage)

def parse_args():
    parser = ArgumentParser(description='Training with DDP.')

    parser.add_argument("--config-file", default="./configs/PSTdataset/swin_v2/swin_v2_tiny.yaml", metavar="FILE", help="path to config file")
    parser.add_argument("--num-gpus", type=int, default=1, help="number of gpus *per machine*")
    parser.add_argument('--work_dir',
                        type=str,
                        default='checkpoints')
    parser.add_argument('--name',
                        type=str,
                        default=None)
    parser.add_argument('--seed',
                        type=int,
                        default=1024)
    parser.add_argument('--checkpoint', 
                        type=str)
    parser.add_argument("--check_val_every_n_epoch", 
                        type=int, 
                        default=5, 
                        help="check_val_every_n_epoch")
    parser.add_argument("--resume_ckpt_path", 
                        default=None, 
                        help="resume_ckpt_path")
    parser.add_argument("--use-semoe-fusion", choices=["true", "false"], default=None)
    parser.add_argument("--router-type", choices=["visual", "class_aware"], default=None)
    parser.add_argument("--decoder-type", choices=["baseline", "semantic_query"], default=None)
    parser.add_argument("--recursive-rerouting", choices=["true", "false"], default=None)
    args = parser.parse_args()

    return args


def my_collate_fn(batch_dict):
    return batch_dict

def setup(args):
    """
    Create configs and perform basic setups.
    """
    cfg = get_cfg() 
    add_deeplab_config(cfg)
    add_maskformer2_config(cfg)
    add_peafusion_config(cfg)
    cfg.merge_from_file(args.config_file)

    cfg.defrost()
    if args.use_semoe_fusion is not None:
        cfg.MODEL.FUSION.USE_SEMOE_FUSION = args.use_semoe_fusion == "true"
        cfg.MODEL.FUSION.FUSION_TYPE = "semoe" if cfg.MODEL.FUSION.USE_SEMOE_FUSION else "peafusion"
    if args.router_type is not None:
        cfg.MODEL.FUSION.ROUTER_TYPE = args.router_type
    if args.decoder_type is not None:
        cfg.MODEL.MASK_FORMER.DECODER_TYPE = args.decoder_type
    if args.recursive_rerouting is not None:
        cfg.MODEL.MASK_FORMER.RECURSIVE_REROUTING = args.recursive_rerouting == "true"

    cfg.freeze()
    return cfg

def main():
    # parse args
    args = parse_args()
    cfg  = setup(args)
    torch.set_float32_matmul_precision("high")
    experiment_name = args.name or osp.splitext(osp.basename(args.config_file))[0]
    print(f'Now training with {args.config_file}...')
    print("=" * 80)
    print(f"Experiment: {experiment_name}")
    print(f"Config: {args.config_file}")
    print(f"Dataset: {cfg.DATASETS.NAME} | Dir: {cfg.DATASETS.DIR}")
    print(f"Batch Size: {cfg.SOLVER.IMS_PER_BATCH} | GPUs: {args.num_gpus} | Seed: {args.seed}")
    print(
        "Fusion: "
        f"use_semoe={cfg.MODEL.FUSION.USE_SEMOE_FUSION}, "
        f"type={cfg.MODEL.FUSION.FUSION_TYPE}, "
        f"router={cfg.MODEL.FUSION.ROUTER_TYPE}, "
        f"class_embed_dim={cfg.MODEL.FUSION.CLASS_EMBED_DIM}, "
        f"expert_depth={cfg.MODEL.FUSION.EXPERT_DEPTH}, "
        f"decoder={cfg.MODEL.MASK_FORMER.DECODER_TYPE}, "
        f"recursive={cfg.MODEL.MASK_FORMER.RECURSIVE_REROUTING}"
    )
    print("=" * 80)

    # configure seed
    seed_everything(args.seed)

    # prepare data loader
    dataset = build_dataset(cfg)

    train_loader = DataLoader(dataset['train'], cfg.SOLVER.IMS_PER_BATCH , shuffle=True, num_workers=cfg.DATASETS.WORKERS_PER_GPU, drop_last=True, collate_fn=my_collate_fn, pin_memory=True )
    val_loader   = DataLoader(dataset['test'], cfg.SOLVER.IMS_PER_BATCH , shuffle=False, num_workers=cfg.DATASETS.WORKERS_PER_GPU, drop_last=False, collate_fn=my_collate_fn, pin_memory=True)

    # define model
    model = MODELS.build(name=cfg.MODEL.META_ARCHITECTURE, option=cfg)

    if args.checkpoint:
        model.load_state_dict(torch.load(args.checkpoint)['state_dict'])

    # define trainer
    work_dir = osp.join(args.work_dir, experiment_name)
    os.makedirs(work_dir, exist_ok=True)
    checkpoint_callback = ModelCheckpoint(dirpath=work_dir,
                                        #   save_weights_only=True,
                                          save_weights_only=False,  # Save the full training state, including optimizer and scheduler
                                          monitor='val/average_IoU',
                                          mode='max',
                                          save_top_k=1,
                                          filename='checkpoint_{epoch:02d}_{step}')
    metrics_logger_callback = EpochMetricsFileLogger(
        osp.join(work_dir, "log.txt"),
        step_log_path=osp.join(work_dir, "train_step_log.txt"),
        experiment_name=experiment_name,
        config_file=args.config_file,
        cfg=cfg,
    )
    csv_logger = CSVLogger(save_dir=work_dir, name="lightning_logs", version="main")
    rich_progress_bar = RichProgressBar(leave=True)
    rich_model_summary = RichModelSummary(max_depth=2)

    trainer_kwargs = dict(
        default_root_dir=work_dir,
        accelerator="gpu" if args.num_gpus > 0 else "cpu",
        devices=args.num_gpus,
        num_nodes=1,
        # max_epochs=cfg.SOLVER.total_epochs,
        max_steps=cfg.SOLVER.MAX_ITER,
        callbacks=[
            checkpoint_callback,
            metrics_logger_callback,
            rich_progress_bar,
            rich_model_summary,
        ],
        logger=csv_logger,
        check_val_every_n_epoch=args.check_val_every_n_epoch,
        enable_progress_bar=True,
        # precision=16
    )
    if args.num_gpus > 1:
        trainer_kwargs["strategy"] = "ddp"

    trainer = Trainer(**trainer_kwargs)

    # training
    trainer.fit(model, train_loader, val_loader, ckpt_path= args.resume_ckpt_path)

if __name__ == '__main__':
    main()
