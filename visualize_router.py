import argparse
import os
import sys
from typing import Dict, Optional

import cv2
import numpy as np
import torch
from PIL import Image

from detectron2.config import get_cfg
from detectron2.projects.deeplab import add_deeplab_config

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from models import MODELS
from models.config import add_peafusion_config
from models.mask2former import add_maskformer2_config
from util.util import visualize_pred


def parse_args():
    parser = argparse.ArgumentParser(description="Visualize SeMoE router weights.")
    parser.add_argument("--checkpoint", required=True, type=str)
    parser.add_argument("--config-file", required=True, type=str)
    parser.add_argument("--rgb-path", required=True, type=str)
    parser.add_argument("--thermal-path", required=True, type=str)
    parser.add_argument("--output-dir", required=True, type=str)
    parser.add_argument("--class-id", "--class_id", dest="class_id", type=int, default=None)
    parser.add_argument("--class-name", "--class_name", dest="class_name", type=str, default=None)
    return parser.parse_args()


def setup_cfg(config_file):
    cfg = get_cfg()
    add_deeplab_config(cfg)
    add_maskformer2_config(cfg)
    add_peafusion_config(cfg)
    cfg.merge_from_file(config_file)
    cfg.defrost()
    pretrained_path = cfg.MODEL.SWIN.PRETRAINED
    if isinstance(pretrained_path, str) and pretrained_path and not os.path.exists(pretrained_path):
        print(f"Backbone pretrained weights not found at {pretrained_path}. Falling back to None.")
        cfg.MODEL.SWIN.PRETRAINED = None
    cfg.freeze()
    return cfg


def read_rgb_image(path):
    image = Image.open(path).convert("RGB")
    return np.array(image)


def read_thermal_image(path):
    image = Image.open(path)
    thermal = np.array(image)
    if thermal.ndim == 3:
        thermal = thermal[..., -1]
    return thermal


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def save_rgb(path, image):
    cv2.imwrite(path, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))


def save_grayscale(path, image):
    if image.ndim == 2:
        image = np.repeat(image[..., None], 3, axis=2)
    cv2.imwrite(path, cv2.cvtColor(image.astype(np.uint8), cv2.COLOR_RGB2BGR))


def tensor_to_constant_map(weight_tensor, out_h, out_w):
    value = float(weight_tensor.mean().item())
    return np.full((out_h, out_w), value, dtype=np.float32)


def normalize_map(weight_map):
    min_val = float(weight_map.min())
    max_val = float(weight_map.max())
    if max_val - min_val < 1e-8:
        return np.zeros_like(weight_map, dtype=np.uint8)
    norm = (weight_map - min_val) / (max_val - min_val)
    return (norm * 255.0).astype(np.uint8)


def save_heatmap(path, weight_map):
    heat_uint8 = normalize_map(weight_map)
    heatmap = cv2.applyColorMap(heat_uint8, cv2.COLORMAP_JET)
    cv2.imwrite(path, heatmap)


def select_class_id(model, pred_mask, class_id, class_name):
    if class_id is not None:
        return class_id
    if class_name is not None:
        if not hasattr(model, "label_list"):
            raise ValueError("Model does not expose label_list, cannot resolve class_name.")
        if class_name not in model.label_list:
            raise ValueError(f"class_name '{class_name}' not found in label list: {model.label_list}")
        return model.label_list.index(class_name)

    values, counts = np.unique(pred_mask, return_counts=True)
    valid = [(v, c) for v, c in zip(values.tolist(), counts.tolist()) if v != 0]
    if len(valid) == 0:
        return int(values[np.argmax(counts)])
    valid.sort(key=lambda item: item[1], reverse=True)
    return int(valid[0][0])


def reduce_router_weights(router_weights, expert_index, class_id=None):
    weights = router_weights
    if isinstance(weights, torch.Tensor):
        weights = weights.detach().cpu()

    if weights.dim() == 6:
        # [B, K, 3, C, 1, 1]
        if class_id is None:
            class_slice = weights[0, :, expert_index].mean(dim=0)
        else:
            class_slice = weights[0, class_id, expert_index]
        return class_slice.mean()
    if weights.dim() == 5:
        # [B, K, 3, 1, 1]
        if class_id is None:
            return weights[0, :, expert_index].mean()
        return weights[0, class_id, expert_index].mean()
    if weights.dim() == 4:
        # [B, 3, C, 1] or [B, C, 3, 1]
        for dim_idx, dim_size in enumerate(weights.shape):
            if dim_size == 3:
                weights = weights.movedim(dim_idx, 1)
                return weights[0, expert_index].mean()
    if weights.dim() == 3:
        # [B, 3, C] or [B, K, 3]
        if weights.shape[1] == 3:
            return weights[0, expert_index].mean()
        return weights[0, :, expert_index].mean()
    if weights.dim() == 2:
        # [B, 3]
        return weights[0, expert_index].mean()
    raise ValueError(f"Unsupported router weight shape: {tuple(weights.shape)}")


def save_router_group(output_dir, group_name, router_weights, out_h, out_w, class_id=None):
    expert_names = ["rgb", "thermal", "shared"]
    for expert_index, expert_name in enumerate(expert_names):
        scalar_map = tensor_to_constant_map(
            reduce_router_weights(router_weights, expert_index, class_id=class_id),
            out_h,
            out_w,
        )
        save_heatmap(
            os.path.join(output_dir, f"{group_name}_alpha_{expert_name}.png"),
            scalar_map,
        )


def main():
    args = parse_args()
    ensure_dir(args.output_dir)

    cfg = setup_cfg(args.config_file)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = MODELS.build(name=cfg.MODEL.META_ARCHITECTURE, option=cfg)
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint["state_dict"])
    model.to(device)
    model.eval()
    model.network.eval()

    rgb = read_rgb_image(args.rgb_path)
    thermal = read_thermal_image(args.thermal_path)
    h, w = rgb.shape[:2]
    thermal_resized = thermal
    if thermal.shape[:2] != (h, w):
        thermal_resized = cv2.resize(thermal, (w, h), interpolation=cv2.INTER_LINEAR)

    image_4ch = np.concatenate([rgb, thermal_resized[..., None]], axis=2).astype(np.float32)
    image_tensor = torch.from_numpy(np.ascontiguousarray(image_4ch.transpose(2, 0, 1))).to(device)
    batched_inputs = [{"image": image_tensor, "height": h, "width": w}]

    with torch.no_grad():
        outputs, _ = model.network(batched_inputs)

    sem_seg = outputs[0]["sem_seg"]
    pred_mask = sem_seg.argmax(0).detach().cpu().numpy().astype(np.uint8)
    pred_vis = visualize_pred(model.palette, pred_mask)

    save_rgb(os.path.join(args.output_dir, "rgb.png"), rgb)
    thermal_vis = np.repeat(thermal_resized[..., None], 3, axis=2).astype(np.uint8)
    save_grayscale(os.path.join(args.output_dir, "thermal.png"), thermal_vis)
    save_rgb(os.path.join(args.output_dir, "segmentation_prediction.png"), pred_vis)

    selected_class_id = select_class_id(model, pred_mask, args.class_id, args.class_name)
    if hasattr(model, "label_list") and selected_class_id < len(model.label_list):
        print(f"Using class_id={selected_class_id}, class_name={model.label_list[selected_class_id]}")
    else:
        print(f"Using class_id={selected_class_id}")

    stage_router_weights: Dict[str, Dict[str, torch.Tensor]] = getattr(
        model.network.backbone, "latest_stage_router_weights", {}
    )
    for stage_name, weight_dict in stage_router_weights.items():
        save_router_group(args.output_dir, stage_name, weight_dict["alpha"], h, w)

    predictor = model.network.sem_seg_head.predictor
    latest_aux_outputs = getattr(predictor, "latest_aux_outputs", [])
    for layer_idx, aux_output in enumerate(latest_aux_outputs):
        if "router_weights" not in aux_output:
            continue
        save_router_group(
            args.output_dir,
            f"decoder_layer_{layer_idx}",
            aux_output["router_weights"],
            h,
            w,
            class_id=selected_class_id,
        )

    print(f"Saved visualization outputs to {args.output_dir}")


if __name__ == "__main__":
    main()
