#!/usr/bin/env python3
import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import torch


DATASET_SPECS = {
    "pst": [
        {
            "label": "unlabeled",
            "aliases": ["background", "unlabeled region", "background region"],
            "background": True,
        },
        {"label": "fire_extinhuisher", "aliases": ["fire extinguisher"]},
        {"label": "backpack", "aliases": ["backpack"]},
        {"label": "hand_drill", "aliases": ["hand drill", "power drill"]},
        {"label": "rescue_randy", "aliases": ["survivor", "rescue dummy", "rescue randy mannequin"]},
    ],
    "fmb": [
        {
            "label": "unlabeled",
            "aliases": ["background", "unlabeled region", "background region"],
            "background": True,
        },
        {"label": "Road", "aliases": ["road"]},
        {"label": "Sidewalk", "aliases": ["sidewalk", "pavement"]},
        {"label": "Building", "aliases": ["building"]},
        {"label": "Traffic Lamp", "aliases": ["traffic light", "traffic lamp"]},
        {"label": "Traffic Sign", "aliases": ["traffic sign", "road sign"]},
        {"label": "Vegetation", "aliases": ["vegetation", "plants"]},
        {"label": "Sky", "aliases": ["sky"]},
        {"label": "Person", "aliases": ["person", "pedestrian"]},
        {"label": "Car", "aliases": ["car"]},
        {"label": "Truck", "aliases": ["truck"]},
        {"label": "Bus", "aliases": ["bus"]},
        {"label": "Motorcycle", "aliases": ["motorcycle"]},
        {"label": "Bicycle", "aliases": ["bicycle", "bike"]},
        {"label": "Pole", "aliases": ["pole", "utility pole"]},
    ],
    "mfnet": [
        {
            "label": "unlabeled",
            "aliases": ["background", "unlabeled region", "background region"],
            "background": True,
        },
        {"label": "car", "aliases": ["car"]},
        {"label": "person", "aliases": ["person", "pedestrian"]},
        {"label": "bike", "aliases": ["bicycle", "bike"]},
        {"label": "curve", "aliases": ["road curve", "curved road"]},
        {"label": "car_stop", "aliases": ["car stop", "parking barrier", "road blocker"]},
        {"label": "guardrail", "aliases": ["guardrail", "road guardrail"]},
        {"label": "color_cone", "aliases": ["traffic cone", "color cone", "road cone"]},
        {"label": "bump", "aliases": ["speed bump", "road bump"]},
    ],
}


def build_prompts(class_spec):
    if class_spec.get("background", False):
        return [
            "background",
            "background region",
            "unlabeled region",
            "a thermal-visible scene background",
            "an infrared scene background",
        ]

    templates = [
        "{name}",
        "a thermal image of {name}",
        "an infrared image of {name}",
        "a thermal-visible image of {name}",
        "a multispectral image of {name}",
    ]
    prompts = []
    for alias in class_spec["aliases"]:
        for template in templates:
            prompts.append(template.format(name=alias))
    # Keep order stable while removing duplicates.
    seen = set()
    deduped = []
    for prompt in prompts:
        if prompt not in seen:
            deduped.append(prompt)
            seen.add(prompt)
    return deduped


def load_languagebind_thermal(languagebind_root, ckpt_dir):
    sys.path.insert(0, str(languagebind_root))
    from languagebind import LanguageBindThermal, LanguageBindThermalTokenizer

    model = LanguageBindThermal.from_pretrained(str(ckpt_dir), local_files_only=True)
    tokenizer = LanguageBindThermalTokenizer.from_pretrained(str(ckpt_dir), local_files_only=True)
    return model, tokenizer


def encode_prompt_set(model, tokenizer, prompts, device):
    batch = tokenizer(
        prompts,
        max_length=77,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    )
    batch = {k: v.to(device) for k, v in batch.items()}
    with torch.no_grad():
        text_features = model.get_text_features(**batch).float()
    text_features = text_features / text_features.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    prototype = text_features.mean(dim=0)
    prototype = prototype / prototype.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    return prototype.cpu()


def generate_dataset_embeddings(model, tokenizer, dataset_name, device):
    specs = DATASET_SPECS[dataset_name]
    embeddings = []
    metadata = []
    for class_id, class_spec in enumerate(specs):
        prompts = build_prompts(class_spec)
        prototype = encode_prompt_set(model, tokenizer, prompts, device)
        embeddings.append(prototype)
        metadata.append(
            {
                "class_id": class_id,
                "label": class_spec["label"],
                "aliases": class_spec["aliases"],
                "prompts": prompts,
            }
        )
    embeddings = torch.stack(embeddings, dim=0)
    return embeddings, metadata


def save_outputs(output_dir, dataset_name, embeddings, metadata, ckpt_dir, device):
    output_dir.mkdir(parents=True, exist_ok=True)
    tensor_path = output_dir / f"{dataset_name}_class_embedding.pt"
    meta_path = output_dir / f"{dataset_name}_class_embedding.meta.json"
    torch.save(embeddings, tensor_path)
    with meta_path.open("w", encoding="utf-8") as f:
        json.dump(
            {
                "dataset": dataset_name,
                "shape": list(embeddings.shape),
                "dtype": str(embeddings.dtype),
                "normalized": True,
                "model": "LanguageBind_Thermal",
                "checkpoint_dir": str(ckpt_dir),
                "device_used": str(device),
                "created_at": datetime.now().isoformat(timespec="seconds"),
                "classes": metadata,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )
    return tensor_path, meta_path


def parse_args():
    parser = argparse.ArgumentParser(description="Generate PST/FMB/MFNet class embeddings with LanguageBind_Thermal.")
    parser.add_argument(
        "--languagebind-root",
        type=Path,
        default=Path("/home/wislab/lzh/LanguageBind-main"),
        help="Path to the LanguageBind code repository.",
    )
    parser.add_argument(
        "--ckpt-dir",
        type=Path,
        default=Path("/home/wislab/lzh/LanguageBind-main/pretrained_model/LanguageBind_Thermal"),
        help="Local LanguageBind_Thermal checkpoint directory.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("/home/wislab/lzh/PEAFusion-main/cls_embed"),
        help="Directory to save generated class embeddings.",
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=["mfnet", "fmb", "pst"],
        choices=sorted(DATASET_SPECS.keys()),
        help="Datasets to generate embeddings for.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Device to use: auto / cpu / cuda / cuda:0 ...",
    )
    return parser.parse_args()


def resolve_device(device_arg):
    if device_arg != "auto":
        return torch.device(device_arg)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main():
    args = parse_args()
    if not args.languagebind_root.exists():
        raise FileNotFoundError(f"LanguageBind root not found: {args.languagebind_root}")
    if not args.ckpt_dir.exists():
        raise FileNotFoundError(f"Checkpoint directory not found: {args.ckpt_dir}")

    device = resolve_device(args.device)
    print(f"Loading LanguageBind_Thermal from: {args.ckpt_dir}")
    print(f"Using device: {device}")
    model, tokenizer = load_languagebind_thermal(args.languagebind_root, args.ckpt_dir)
    model = model.to(device)
    model.eval()

    for dataset_name in args.datasets:
        embeddings, metadata = generate_dataset_embeddings(model, tokenizer, dataset_name, device)
        tensor_path, meta_path = save_outputs(
            args.output_dir, dataset_name, embeddings, metadata, args.ckpt_dir, device
        )
        print(
            f"[{dataset_name}] shape={tuple(embeddings.shape)} "
            f"saved_tensor={tensor_path} saved_meta={meta_path}"
        )


if __name__ == "__main__":
    main()
