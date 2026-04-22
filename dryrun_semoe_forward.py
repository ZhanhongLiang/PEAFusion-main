"""
Dry-run script for PEAFusion with SeMoE fusion enabled.

This script does not touch the training pipeline. It only builds the model with
random inputs and verifies that the segmentation forward pass runs without shape
errors.
"""

from __future__ import annotations

import os
import sys

import torch
from detectron2.config import get_cfg
from detectron2.projects.deeplab import add_deeplab_config

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from models.config import add_peafusion_config
from models.mask2former import RGBTMaskFormer, add_maskformer2_config


def build_cfg():
    cfg = get_cfg()
    add_deeplab_config(cfg)
    add_maskformer2_config(cfg)
    add_peafusion_config(cfg)
    cfg.merge_from_file(os.path.join(PROJECT_ROOT, "configs/PSTdataset/swin_v2/swin_v2_tiny.yaml"))

    cfg.defrost()
    cfg.MODEL.SWIN.PRETRAINED = None
    cfg.MODEL.SWIN.MODEL_OUTPUT_ATTN = False
    cfg.MODEL.FUSION.USE_SEMOE_FUSION = True
    cfg.MODEL.FUSION.FUSION_TYPE = "semoe"
    cfg.MODEL.FUSION.SEMOE_CHANNEL_WISE = True
    cfg.freeze()
    return cfg


def main():
    cfg = build_cfg()
    model = RGBTMaskFormer(cfg)
    model.eval()

    image = torch.randn(4, 64, 64)
    batched_inputs = [{"image": image}]

    with torch.no_grad():
        outputs, _ = model(batched_inputs)

    sem_seg = outputs[0]["sem_seg"]
    print("segmentation logits shape:", tuple(sem_seg.shape))


if __name__ == "__main__":
    main()
