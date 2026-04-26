# coding:utf-8
# Modified by Yan Wang based on the following repositories.
# CRM_RGBTSeg: https://github.com/UkcheolShin/CRM_RGBTSeg

import os
import copy
import itertools
import pickle
import re

import torch
import torch.nn as nn 
from torch.nn import functional as F
from pytorch_lightning import LightningModule
import torchvision.utils as vutils

from torchmetrics.classification import MulticlassPrecision, MulticlassRecall, MulticlassJaccardIndex

from util.util import compute_results, get_palette_MF, get_palette_PST, get_palette_FMB ,visualize_pred
from .registry import MODELS
from models.mask2former import RGBTMaskFormer
from detectron2.projects.deeplab import build_lr_scheduler
from detectron2.solver.build import maybe_add_gradient_clipping
import cv2
import numpy as np

@MODELS.register_module(name='RGBTMaskFormer')
class Model_RGBT_Mask2Former(LightningModule):
    def __init__(self, cfg):
        super(Model_RGBT_Mask2Former, self).__init__()
        self.save_hyperparameters()

        self.cfg = cfg
        self.num_classes = cfg.MODEL.SEM_SEG_HEAD.NUM_CLASSES
        self.ignore_label = cfg.MODEL.SEM_SEG_HEAD.IGNORE_VALUE
        self.learning_rate = cfg.SOLVER.BASE_LR
        self.lr_decay = cfg.SOLVER.WEIGHT_DECAY

        if self.num_classes == 9 : 
            self.label_list = ["unlabeled", "car", "person", "bike", "curve", "car_stop", "guardrail", "color_cone", "bump"]
            self.palette = get_palette_MF()
            self.val_precision = MulticlassPrecision(num_classes=self.num_classes,average=None, dist_sync_on_step=True)
            self.val_recall = MulticlassRecall(num_classes=self.num_classes,average=None, dist_sync_on_step=True)
            self.val_iou = MulticlassJaccardIndex(num_classes=self.num_classes,average=None, dist_sync_on_step=True)
        elif self.num_classes == 5 : 
            self.label_list = ["unlabeled", "fire_extinhuisher", "backpack", "hand_drill", "rescue_randy"]
            self.palette = get_palette_PST()
            self.val_precision = MulticlassPrecision(num_classes=self.num_classes,average=None, dist_sync_on_step=True)
            self.val_recall = MulticlassRecall(num_classes=self.num_classes,average=None, dist_sync_on_step=True)
            self.val_iou = MulticlassJaccardIndex(num_classes=self.num_classes,average=None, dist_sync_on_step=True)
        elif self.num_classes == 15 : 
            self.label_list = ["unlabeled","Road", "Sidewalk", "Building", "Traffic Lamp", "Traffic Sign", "Vegetation", 
                    "Sky", "Person", "Car", "Truck", "Bus", "Motorcycle", "Bicycle", "Pole"] 
            self.palette = get_palette_FMB()      
            self.val_precision = MulticlassPrecision(num_classes=self.num_classes,average=None, dist_sync_on_step=True, ignore_index= 0)
            self.val_recall = MulticlassRecall(num_classes=self.num_classes, average=None,dist_sync_on_step=True, ignore_index=0)
            self.val_iou = MulticlassJaccardIndex(num_classes=self.num_classes,average=None, dist_sync_on_step=True, ignore_index=0)

        self.network = RGBTMaskFormer(cfg)
        self.use_class_probe_loss = cfg.MODEL.MASK_FORMER.USE_CLASS_PROBE_LOSS
        self.class_probe_loss_weight = cfg.MODEL.MASK_FORMER.LOSS_CLASS_PROBE_WEIGHT
        if self.use_class_probe_loss:
            if (
                not cfg.MODEL.FUSION.USE_SEMOE_FUSION
                or cfg.MODEL.FUSION.ROUTER_TYPE != "class_aware"
                or not cfg.MODEL.FUSION.CLASS_INDEPENDENT
            ):
                raise ValueError(
                    "USE_CLASS_PROBE_LOSS requires SeMoE fusion with ROUTER_TYPE='class_aware' "
                    "and MODEL.FUSION.CLASS_INDEPENDENT=True."
                )
            probe_in_channels = cfg.MODEL.SWIN.EMBED_DIM
            self.network.class_probe_head = nn.Conv2d(
                probe_in_channels, 1, kernel_size=1, stride=1, padding=0, bias=True
            )
            nn.init.xavier_uniform_(self.network.class_probe_head.weight)
            nn.init.constant_(self.network.class_probe_head.bias, 0.0)
        else:
            self.network.class_probe_head = None
        self.optimizer = self.build_optimizer(cfg, self.network)
        self.scheduler = self.build_lr_scheduler(cfg, self.optimizer)
        self.automatic_optimization = False

        self.balance_loss_weight = cfg.MODEL.MASK_FORMER.LOSS_BALANCE_WEIGHT
        self.consistency_loss_weight = cfg.MODEL.MASK_FORMER.LOSS_CONSISTENCY_WEIGHT
        self.aux_loss_weight = cfg.MODEL.MASK_FORMER.LOSS_AUX_WEIGHT
        self.use_consistency_loss = cfg.MODEL.MASK_FORMER.USE_CONSISTENCY_LOSS
        self.val_confmat = torch.zeros(
            (self.num_classes, self.num_classes), dtype=torch.long
        )

    def _prepare_valid_eval_pairs(self, pred: torch.Tensor, label: torch.Tensor):
        pred = pred.long().reshape(-1)
        label = label.long().reshape(-1)
        valid = label != self.ignore_label
        valid = valid & (label >= 0) & (label < self.num_classes)
        valid = valid & (pred >= 0) & (pred < self.num_classes)
        return pred[valid], label[valid]

    def _update_confmat(self, pred: torch.Tensor, label: torch.Tensor):
        if pred.numel() == 0:
            return
        indices = label * self.num_classes + pred
        confmat = torch.bincount(indices, minlength=self.num_classes ** 2).reshape(
            self.num_classes, self.num_classes
        )
        self.val_confmat = self.val_confmat.to(confmat.device)
        self.val_confmat += confmat.to(self.val_confmat.device)

    @classmethod
    def build_lr_scheduler(cls, cfg, optimizer):
        """
        It now calls :func:`detectron2.solver.build_lr_scheduler`.
        Overwrite it if you'd like a different scheduler.
        """
        return build_lr_scheduler(cfg, optimizer)

    @classmethod
    def build_optimizer(cls, cfg, model):
        weight_decay_norm = cfg.SOLVER.WEIGHT_DECAY_NORM
        weight_decay_embed = cfg.SOLVER.WEIGHT_DECAY_EMBED

        defaults = {}
        defaults["lr"] = cfg.SOLVER.BASE_LR
        defaults["weight_decay"] = cfg.SOLVER.WEIGHT_DECAY

        norm_module_types = (
            torch.nn.BatchNorm1d,
            torch.nn.BatchNorm2d,
            torch.nn.BatchNorm3d,
            torch.nn.SyncBatchNorm,
            # NaiveSyncBatchNorm inherits from BatchNorm2d
            torch.nn.GroupNorm,
            torch.nn.InstanceNorm1d,
            torch.nn.InstanceNorm2d,
            torch.nn.InstanceNorm3d,
            torch.nn.LayerNorm,
            torch.nn.LocalResponseNorm,
        )

        params: List[Dict[str, Any]] = []
        memo: Set[torch.nn.parameter.Parameter] = set()
        for module_name, module in model.named_modules():
            for module_param_name, value in module.named_parameters(recurse=False):
                if not value.requires_grad:
                    continue
                # Avoid duplicating parameters
                if value in memo:
                    continue
                memo.add(value)

                hyperparams = copy.copy(defaults)
                if "backbone" in module_name:
                    hyperparams["lr"] = hyperparams["lr"] * cfg.SOLVER.BACKBONE_MULTIPLIER
                if (
                    "relative_position_bias_table" in module_param_name
                    or "absolute_pos_embed" in module_param_name
                ):
                    print(module_param_name)
                    hyperparams["weight_decay"] = 0.0
                if isinstance(module, norm_module_types):
                    hyperparams["weight_decay"] = weight_decay_norm
                if isinstance(module, torch.nn.Embedding):
                    hyperparams["weight_decay"] = weight_decay_embed
                params.append({"params": [value], **hyperparams})

        def maybe_add_full_model_gradient_clipping(optim):
            # detectron2 doesn't have full model gradient clipping now
            clip_norm_val = cfg.SOLVER.CLIP_GRADIENTS.CLIP_VALUE
            enable = (
                cfg.SOLVER.CLIP_GRADIENTS.ENABLED
                and cfg.SOLVER.CLIP_GRADIENTS.CLIP_TYPE == "full_model"
                and clip_norm_val > 0.0
            )

            class FullModelGradientClippingOptimizer(optim):
                def step(self, closure=None):
                    all_params = itertools.chain(*[x["params"] for x in self.param_groups])
                    torch.nn.utils.clip_grad_norm_(all_params, clip_norm_val)
                    super().step(closure=closure)

            return FullModelGradientClippingOptimizer if enable else optim

        optimizer_type = cfg.SOLVER.OPTIMIZER
        if optimizer_type == "SGD":
            optimizer = maybe_add_full_model_gradient_clipping(torch.optim.SGD)(
                params, cfg.SOLVER.BASE_LR, momentum=cfg.SOLVER.MOMENTUM
            )
        elif optimizer_type == "ADAMW":
            optimizer = maybe_add_full_model_gradient_clipping(torch.optim.AdamW)(
                params, cfg.SOLVER.BASE_LR
            )
        else:
            raise NotImplementedError(f"no optimizer type {optimizer_type}")
        if not cfg.SOLVER.CLIP_GRADIENTS.CLIP_TYPE == "full_model":
            optimizer = maybe_add_gradient_clipping(cfg, optimizer)
        return optimizer

    def configure_optimizers(self):
        optimizer = self.optimizer
        scheduler = self.scheduler
        return [optimizer], [scheduler]

    def forward(self, x):
        logits = self.network(x)
        return logits.argmax(1).squeeze()

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        x = batch
        logits = self.network(x)
        return logits

    def _is_aux_loss_key(self, key):
        return re.search(r"_\d+$", key) is not None

    def _apply_aux_loss_weight(self, losses_dict):
        # Keep baseline behavior unchanged. Only scale decoder auxiliary losses for
        # semantic query decoder where the intermediate logits are used explicitly.
        if self.cfg.MODEL.MASK_FORMER.DECODER_TYPE != "semantic_query":
            return losses_dict
        if self.aux_loss_weight == 1.0:
            return losses_dict

        scaled_losses = {}
        for key, value in losses_dict.items():
            if self._is_aux_loss_key(key):
                scaled_losses[key] = value * self.aux_loss_weight
            else:
                scaled_losses[key] = value
        return scaled_losses

    def _flatten_router_usage(self, router_weights):
        if router_weights.dim() < 2:
            raise ValueError("router_weights must have at least 2 dimensions.")
        if router_weights.shape[-1] == 3:
            expert_dim = router_weights.dim() - 1
        else:
            expert_dim = None
            for dim_idx, dim_size in enumerate(router_weights.shape):
                if dim_size == 3:
                    expert_dim = dim_idx
                    break
            if expert_dim is None:
                raise ValueError("router_weights must contain an expert dimension of size 3.")

        router_weights = router_weights.movedim(expert_dim, -1)
        return router_weights.reshape(-1, 3)

    def _compute_expert_balance_loss(self):
        if self.balance_loss_weight <= 0:
            return None

        backbone = getattr(self.network, "backbone", None)
        if backbone is None:
            return None

        stage_router_weights = getattr(backbone, "latest_stage_router_weights", {})
        if not isinstance(stage_router_weights, dict) or len(stage_router_weights) == 0:
            return None

        collected = []
        for _, stage_output in sorted(stage_router_weights.items()):
            if not isinstance(stage_output, dict):
                continue
            router_weights = stage_output.get("routing_weights")
            if router_weights is None:
                router_weights = stage_output.get("alpha")
            if router_weights is None:
                continue
            flattened_usage = self._flatten_router_usage(router_weights)
            collected.append(flattened_usage.mean(dim=0))

        if len(collected) == 0:
            return None

        usage = torch.stack(collected, dim=0).mean(dim=0)
        target = torch.full_like(usage, 1.0 / 3.0)
        return ((usage - target) ** 2).sum()

    def _background_class_offset(self):
        if not hasattr(self, "label_list") or len(self.label_list) == 0:
            return 0
        first_label = self.label_list[0].lower()
        if "unlabeled" in first_label or "background" in first_label:
            return 1
        return 0

    def _compute_class_probe_loss(self, labels):
        if not self.use_class_probe_loss or self.class_probe_loss_weight <= 0:
            return None

        class_probe_head = getattr(self.network, "class_probe_head", None)
        if class_probe_head is None:
            return None

        backbone = getattr(self.network, "backbone", None)
        if backbone is None:
            return None

        class_features = getattr(backbone, "latest_stage_class_features", {}).get("stage_2")
        if class_features is None:
            return None
        if class_features.dim() != 5:
            raise ValueError(
                f"Expected class-specific features with shape [B, K, C, H, W], got {tuple(class_features.shape)}."
            )

        labels = labels.to(class_features.device).long()
        batch_size, num_classes, channels, height, width = class_features.shape
        if num_classes != self.num_classes:
            raise ValueError(
                f"Expected {self.num_classes} class-specific features, got {num_classes}."
            )

        probe_logits = class_probe_head(
            class_features.reshape(batch_size * num_classes, channels, height, width)
        ).view(batch_size, num_classes, height, width)
        probe_logits = F.interpolate(
            probe_logits,
            size=labels.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )

        valid_pixels = labels != self.ignore_label
        class_ids = torch.arange(num_classes, device=labels.device).view(1, num_classes, 1, 1)
        target_masks = (labels.unsqueeze(1) == class_ids) & valid_pixels.unsqueeze(1)

        valid_mask = valid_pixels.unsqueeze(1).float()
        target_masks = target_masks.float()

        bce_map = F.binary_cross_entropy_with_logits(
            probe_logits, target_masks, reduction="none"
        )
        valid_count = valid_mask.flatten(2).sum(dim=-1).clamp_min(1.0)
        bce_per_class = (bce_map * valid_mask).flatten(2).sum(dim=-1) / valid_count

        pred_prob = torch.sigmoid(probe_logits) * valid_mask
        target_prob = target_masks * valid_mask
        intersection = (pred_prob * target_prob).flatten(2).sum(dim=-1)
        denominator = pred_prob.flatten(2).sum(dim=-1) + target_prob.flatten(2).sum(dim=-1)
        dice_per_class = 1.0 - (2.0 * intersection + 1.0) / (denominator + 1.0)

        present_classes = target_masks.flatten(2).sum(dim=-1) > 0
        present_classes[:, : self._background_class_offset()] = False

        if not present_classes.any():
            return None

        probe_loss = (bce_per_class + dice_per_class)[present_classes].mean()
        return probe_loss

    def _forward_semantic_logits(self, batch_data, drop_mode=None, no_grad=False):
        batch_copy = []
        for sample in batch_data:
            sample_copy = dict(sample)
            image = sample["image"].clone()
            if drop_mode == "rgb":
                image[:3] = 0
            elif drop_mode == "thermal":
                image[3:] = 0
            sample_copy["image"] = image
            batch_copy.append(sample_copy)

        was_training = self.network.training
        self.network.eval()
        context = torch.no_grad() if no_grad else torch.enable_grad()
        with context:
            outputs, _ = self.network(batch_copy)
            logits = torch.stack([x["sem_seg"] for x in outputs], dim=0)
        self.network.train(was_training)
        return logits

    def _compute_consistency_loss(self, batch_data):
        if not self.use_consistency_loss or self.consistency_loss_weight <= 0:
            return None

        drop_mode = "rgb" if torch.rand(1).item() < 0.5 else "thermal"
        teacher_logits = self._forward_semantic_logits(batch_data, drop_mode=None, no_grad=True)
        student_logits = self._forward_semantic_logits(batch_data, drop_mode=drop_mode, no_grad=False)

        teacher_prob = teacher_logits.softmax(dim=1).detach()
        student_log_prob = F.log_softmax(student_logits, dim=1)
        return F.kl_div(student_log_prob, teacher_prob, reduction="batchmean")

    def training_step(self, batch_data, batch_idx):
        # optimizers
        optim = self.optimizers()

        # get input & gt_label
        labels = [x["sem_seg_gt"] for x in batch_data]
        labels = torch.stack(labels) 

        # tensorboard logger
        logger = self.logger.experiment

        # get network output
        losses_dict, attention_maps = self.network(batch_data)
        losses_dict = self._apply_aux_loss_weight(losses_dict)
        maskformer_loss = sum(losses_dict.values())

        loss_balance_value = maskformer_loss.new_zeros(())
        loss_class_probe_value = maskformer_loss.new_zeros(())

        loss_balance = self._compute_expert_balance_loss()
        if loss_balance is not None:
            loss_balance_value = loss_balance * self.balance_loss_weight
            losses_dict["loss_balance"] = loss_balance_value

        loss_consistency = self._compute_consistency_loss(batch_data)
        if loss_consistency is not None:
            losses_dict["loss_consistency"] = loss_consistency * self.consistency_loss_weight

        loss_class_probe = self._compute_class_probe_loss(labels)
        if loss_class_probe is not None:
            loss_class_probe_value = loss_class_probe * self.class_probe_loss_weight
            losses_dict["loss_class_probe"] = loss_class_probe_value

        loss = sum(losses_dict.values()) 

        # optimize network
        optim.zero_grad()
        self.manual_backward(loss)
        optim.step()

        # log
        self.log('train/total_loss', loss, prog_bar=True, on_step=True, on_epoch=True, sync_dist=True)
        self.log('train/maskformer_loss', maskformer_loss.detach(), prog_bar=False, on_step=True, on_epoch=True, sync_dist=True)
        self.log('train/loss_balance', loss_balance_value.detach(), prog_bar=False, on_step=True, on_epoch=True, sync_dist=True)
        self.log('train/loss_class_probe', loss_class_probe_value.detach(), prog_bar=False, on_step=True, on_epoch=True, sync_dist=True)
        for key, value in losses_dict.items():
            if key in {"loss_balance", "loss_class_probe"}:
                continue
            self.log(f"train/{key}", value.detach(), on_step=False, on_epoch=True, sync_dist=True)

        return loss

    def on_train_epoch_end(self):
        scheduler = self.lr_schedulers()
        scheduler.step()

    def validation_step(self, batch_data, batch_idx):
        # get input & gt_label
        images = [x["image"] for x in batch_data]
        labels = [x["sem_seg_gt"] for x in batch_data]

        # tensorboard logger
        logger = self.logger.experiment

        # get network output
        logits, attention_maps = self.network(batch_data)
        logits = [x["sem_seg"] for x in logits]

        images = torch.stack(images) 
        labels = torch.stack(labels) 
        logits = torch.stack(logits) 

        # evaluate performance
        pred  = logits.argmax(1)
        pred, label = self._prepare_valid_eval_pairs(pred, labels)

        # update metrics
        if pred.numel() > 0:
            self.val_precision.update(pred, label)
            self.val_recall.update(pred, label)
            self.val_iou.update(pred, label)
            self._update_confmat(pred, label)


    def on_validation_epoch_end(self):

        precision = self.val_precision.compute()
        recall = self.val_recall.compute()
        iou = self.val_iou.compute()
        confmat = self.val_confmat.float()
        gt_per_class = confmat.sum(dim=1)
        correct_per_class = confmat.diag()
        class_acc = correct_per_class / gt_per_class.clamp_min(1.0)
        overall_acc = correct_per_class.sum() / confmat.sum().clamp_min(1.0)

        if self.num_classes == 15 :  # For FMB dataset
            ignore_indices = torch.tensor([0, 13]) # ignore unlabeled class and bicycle class (bicycle doesn't appear in test set)
            valid_indices = torch.arange(precision.size(0))

            mask = ~torch.any(valid_indices[:, None] == ignore_indices, dim=1)

            valid_precision = precision[mask]
            valid_recall = recall[mask]
            valid_iou = iou[mask]
            valid_acc = class_acc[mask]

            self.log('val/average_precision', valid_precision.mean(), sync_dist=True)  # set sync_dist = True when training via multi-gpu setting.
            self.log('val/average_recall', valid_recall.mean(), sync_dist=True)
            self.log('val/average_IoU', valid_iou.mean(), prog_bar=True, sync_dist=True)
            self.log('val/allAcc', overall_acc, sync_dist=True)
        else:
            self.log('val/average_precision', precision.mean(), sync_dist=True)
            self.log('val/average_recall', recall.mean(), sync_dist=True)
            self.log('val/average_IoU', iou.mean(), prog_bar=True, sync_dist=True)
            self.log('val/allAcc', overall_acc, sync_dist=True)

        assert len(self.label_list) == len(precision), "label_list length must match the number of classes"
        for i in range(len(precision)):
            self.log(f"val(class)/precision_class_{self.label_list[i]}", precision[i].item(), sync_dist=True)
            self.log(f"val(class)/recall_class_{self.label_list[i]}", recall[i].item(), sync_dist=True)
            self.log(f"val(class)/Iou_{self.label_list[i]}", iou[i].item(), sync_dist=True)

        self.val_precision.reset()
        self.val_recall.reset()
        self.val_iou.reset()
        self.val_confmat.zero_()


    def test_step(self, batch_data, batch_idx):
        images = [x["image"] for x in batch_data]
        labels = [x["sem_seg_gt"] for x in batch_data]

        # get network output
        logits, attention_maps = self.network(batch_data)
        logits = [x["sem_seg"] for x in logits]

        images = torch.stack(images) 
        labels = torch.stack(labels) 
        logits = torch.stack(logits) 

        # evaluate performance
        pred  = logits.argmax(1)
        pred, label = self._prepare_valid_eval_pairs(pred, labels)

        # update metrics
        if pred.numel() > 0:
            self.val_precision.update(pred, label)
            self.val_recall.update(pred, label)
            self.val_iou.update(pred, label)

        # save the results
        pred_vis  = visualize_pred(self.palette, logits.argmax(1).squeeze().detach().cpu())
        png_path = os.path.join(self.cfg.SAVE.DIR_ROOT, self.cfg.SAVE.DIR_NAME, "{:05}.png".format(batch_idx))
        cv2.imwrite(png_path, cv2.cvtColor(pred_vis, cv2.COLOR_RGB2BGR))

        # save the attention maps
        if self.cfg.MODEL.SWIN.MODEL_OUTPUT_ATTN:
            attn_dir = os.path.join(self.cfg.SAVE.DIR_ROOT, self.cfg.SAVE.ATTN_DIR_NAME)
            os.makedirs(attn_dir, exist_ok=True)  # Create only the directory structure

            attn_path = os.path.join(attn_dir, "{:05}.npy".format(batch_idx))
            attention_maps = attention_maps.cpu().numpy()
            np.save(attn_path, attention_maps)

        if self.cfg.SAVE.FLAG_VIS_GT:
            # denormalize input images
            images = images.squeeze().detach().cpu().numpy().transpose(1,2,0)
            rgb_vis = images[:,:,:3].astype(np.uint8)
            thr_vis = np.repeat(images[:,:,[-1]], 3, axis=2).astype(np.uint8)
            label_vis = visualize_pred(self.palette, labels.squeeze().detach().cpu())

            png_path = os.path.join(self.cfg.SAVE.DIR_ROOT, "rgb", "{:05}.png".format(batch_idx))
            cv2.imwrite(png_path, cv2.cvtColor(rgb_vis, cv2.COLOR_RGB2BGR))

            png_path = os.path.join(self.cfg.SAVE.DIR_ROOT, "thr", "{:05}.png".format(batch_idx))
            cv2.imwrite(png_path, cv2.cvtColor(thr_vis, cv2.COLOR_RGB2BGR))

            png_path = os.path.join(self.cfg.SAVE.DIR_ROOT, "gt", "{:05}.png".format(batch_idx))
            cv2.imwrite(png_path, cv2.cvtColor(label_vis, cv2.COLOR_RGB2BGR))


    def on_test_epoch_end(self):
        return self.on_validation_epoch_end()
