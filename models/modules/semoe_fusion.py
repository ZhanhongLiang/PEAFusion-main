"""
Experimental SeMoE-Fusion building blocks for RGB-T semantic segmentation.

This module is intentionally standalone and does not modify the original
PEAFusion training or inference pipeline. It provides a lightweight skeleton
for semantic-aware mixture-of-experts fusion on 4D feature maps.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

import torch
import torch.nn as nn


def _validate_expert_depth(depth: int) -> int:
    depth = int(depth)
    if depth < 1:
        raise ValueError(f"expert_depth must be >= 1, got {depth}.")
    return depth


def _load_class_embedding_tensor(
    embedding_path: Union[str, Path],
    num_classes: int,
    embed_dim: int,
) -> Tuple[torch.Tensor, Dict[str, Any]]:
    path = Path(embedding_path)
    expected_shape = (num_classes, embed_dim)
    report: Dict[str, Any] = {
        "status": "not_loaded",
        "success": False,
        "path": str(path),
        "expected_shape": list(expected_shape),
        "meta_found": False,
    }
    if not path.is_file():
        raise FileNotFoundError(f"class embedding file not found: {path}")

    payload = torch.load(path, map_location="cpu")
    if torch.is_tensor(payload):
        tensor = payload
    elif isinstance(payload, dict):
        tensor = None
        for key in ("embeddings", "class_embeddings", "weight", "tensor"):
            value = payload.get(key)
            if torch.is_tensor(value):
                tensor = value
                break
        if tensor is None and len(payload) == 1:
            only_value = next(iter(payload.values()))
            if torch.is_tensor(only_value):
                tensor = only_value
        if tensor is None:
            raise TypeError(
                f"Unsupported class embedding payload in {path}. Expected a tensor or a dict containing a tensor."
            )
    else:
        raise TypeError(f"Unsupported class embedding file content in {path}: {type(payload)}")

    tensor = tensor.detach().float().cpu().contiguous()
    report["loaded_shape"] = list(tensor.shape)
    report["loaded_dtype"] = str(tensor.dtype)
    if tensor.ndim != 2:
        raise ValueError(
            f"class embedding tensor must be 2D [num_classes, embed_dim], got shape {tuple(tensor.shape)} from {path}"
        )
    if tuple(tensor.shape) != expected_shape:
        raise ValueError(
            f"class embedding shape mismatch for {path}: expected {expected_shape}, got {tuple(tensor.shape)}"
        )

    meta_path = path.with_suffix(".meta.json")
    if meta_path.is_file():
        report["meta_found"] = True
        try:
            with meta_path.open("r", encoding="utf-8") as f:
                meta = json.load(f)
            classes = meta.get("classes", [])
            report["meta_class_count"] = len(classes)
            report["meta_labels"] = [item.get("label", "") for item in classes]
            report["meta_labels_preview"] = [item.get("label", "") for item in classes[: min(5, len(classes))]]
        except Exception as exc:  # pragma: no cover - best-effort metadata parsing
            report["meta_error"] = str(exc)

    report["status"] = "loaded"
    report["success"] = True
    report["row_norm_mean"] = float(tensor.norm(dim=1).mean().item())
    return tensor, report


class _ResidualExpertCore(nn.Module):
    """Lightweight residual expert used by the modality-specific experts."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.proj_in = nn.Conv2d(channels, channels, kernel_size=1, bias=True)
        self.act = nn.GELU()
        self.depthwise = nn.Conv2d(
            channels,
            channels,
            kernel_size=3,
            padding=1,
            groups=channels,
            bias=True,
        )
        self.proj_out = nn.Conv2d(channels, channels, kernel_size=1, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.proj_in(x)
        x = self.act(x)
        x = self.depthwise(x)
        x = self.proj_out(x)
        return x + residual


class _ResidualExpertStack(nn.Module):
    """Stacks multiple lightweight residual expert cores."""

    def __init__(self, channels: int, depth: int) -> None:
        super().__init__()
        depth = _validate_expert_depth(depth)
        self.blocks = nn.Sequential(*[_ResidualExpertCore(channels) for _ in range(depth)])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.blocks(x)


class RGBExpert(nn.Module):
    """Enhances RGB features while preserving the input shape."""

    def __init__(self, channels: int, depth: int = 1) -> None:
        super().__init__()
        self.depth = _validate_expert_depth(depth)
        self.core = _ResidualExpertStack(channels, self.depth)

    def forward(self, rgb_feat: torch.Tensor) -> torch.Tensor:
        return self.core(rgb_feat)


class ThermalExpert(nn.Module):
    """Enhances thermal features with parameters independent from RGBExpert."""

    def __init__(self, channels: int, depth: int = 1) -> None:
        super().__init__()
        self.depth = _validate_expert_depth(depth)
        self.core = _ResidualExpertStack(channels, self.depth)

    def forward(self, thermal_feat: torch.Tensor) -> torch.Tensor:
        return self.core(thermal_feat)


class SharedExpert(nn.Module):
    """Builds a shared cross-modal feature from RGB and thermal inputs."""

    def __init__(self, channels: int, depth: int = 1) -> None:
        super().__init__()
        self.depth = _validate_expert_depth(depth)
        fused_channels = channels * 2
        self.reduce = nn.Conv2d(fused_channels, channels, kernel_size=1, bias=True)
        self.act = nn.GELU()
        self.depthwise = nn.Conv2d(
            channels,
            channels,
            kernel_size=3,
            padding=1,
            groups=channels,
            bias=True,
        )
        self.proj = nn.Conv2d(channels, channels, kernel_size=1, bias=True)
        self.refine = (
            nn.Identity()
            if self.depth == 1
            else _ResidualExpertStack(channels, self.depth - 1)
        )

    def forward(self, rgb_feat: torch.Tensor, thermal_feat: torch.Tensor) -> torch.Tensor:
        x = torch.cat([rgb_feat, thermal_feat], dim=1)
        x = self.reduce(x)
        x = self.act(x)
        x = self.depthwise(x)
        x = self.proj(x)
        x = self.refine(x)
        return x


class SemanticExpertRouter(nn.Module):
    """
    Predicts expert weights for RGB, thermal, and shared experts.

    channel_wise=True:
        returns weights of shape [B, 3, C, 1, 1]
    channel_wise=False:
        returns weights of shape [B, 3, 1, 1, 1]
    """

    def __init__(
        self,
        channels: int,
        hidden_ratio: float = 0.25,
        channel_wise: bool = True,
    ) -> None:
        super().__init__()
        hidden_dim = max(8, int(channels * hidden_ratio))
        out_dim = channels * 3 if channel_wise else 3

        self.channels = channels
        self.channel_wise = channel_wise
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.mlp = nn.Sequential(
            nn.Conv2d(channels, hidden_dim, kernel_size=1, bias=True),
            nn.GELU(),
            nn.Conv2d(hidden_dim, out_dim, kernel_size=1, bias=True),
        )

    def forward(self, fused_feat: torch.Tensor) -> torch.Tensor:
        if fused_feat.dim() != 4:
            raise ValueError(f"Expected a 4D tensor [B, C, H, W], got {tuple(fused_feat.shape)}.")
        if fused_feat.shape[1] != self.channels:
            raise ValueError(
                f"Expected input with {self.channels} channels, got {fused_feat.shape[1]}."
            )

        pooled = self.pool(fused_feat)
        logits = self.mlp(pooled)

        if self.channel_wise:
            logits = logits.view(fused_feat.shape[0], 3, self.channels, 1, 1)
            return torch.softmax(logits, dim=1)

        logits = logits.view(fused_feat.shape[0], 3, 1, 1, 1)
        return torch.softmax(logits, dim=1)


class SeMoEFusionBlock(nn.Module):
    """Semantic-aware mixture-of-experts fusion for RGB-T feature maps."""

    def __init__(
        self,
        channels: int,
        channel_wise_router: bool = True,
        return_router_weights: bool = False,
        expert_depth: int = 1,
    ) -> None:
        super().__init__()
        self.channels = channels
        self.expert_depth = _validate_expert_depth(expert_depth)
        self.return_router_weights = return_router_weights
        self.latest_router_weights = {}

        self.rgb_expert = RGBExpert(channels, depth=self.expert_depth)
        self.thermal_expert = ThermalExpert(channels, depth=self.expert_depth)
        self.shared_expert = SharedExpert(channels, depth=self.expert_depth)
        self.router = SemanticExpertRouter(
            channels=channels,
            channel_wise=channel_wise_router,
        )

    def _build_router_input(
        self,
        rgb_feat: torch.Tensor,
        thermal_feat: torch.Tensor,
        shared_feat: torch.Tensor,
        router_feat: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if router_feat is not None:
            return router_feat
        return (rgb_feat + thermal_feat + shared_feat) / 3.0

    def forward(
        self,
        rgb_feat: torch.Tensor,
        thermal_feat: torch.Tensor,
        router_feat: Optional[torch.Tensor] = None,
        return_router_weights: Optional[bool] = None,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict[str, torch.Tensor]]]:
        if rgb_feat.dim() != 4 or thermal_feat.dim() != 4:
            raise ValueError("SeMoEFusionBlock expects 4D inputs [B, C, H, W].")
        if rgb_feat.shape != thermal_feat.shape:
            raise ValueError(
                f"RGB and thermal features must have the same shape, got "
                f"{tuple(rgb_feat.shape)} and {tuple(thermal_feat.shape)}."
            )
        if rgb_feat.shape[1] != self.channels:
            raise ValueError(
                f"Expected input channels to be {self.channels}, got {rgb_feat.shape[1]}."
            )

        rgb_e = self.rgb_expert(rgb_feat)
        thermal_e = self.thermal_expert(thermal_feat)
        shared_e = self.shared_expert(rgb_feat, thermal_feat)

        router_input = self._build_router_input(rgb_e, thermal_e, shared_e, router_feat)
        alpha = self.router(router_input)

        alpha_rgb = alpha[:, 0]
        alpha_t = alpha[:, 1]
        alpha_shared = alpha[:, 2]

        self.latest_router_weights = {
            "alpha": alpha,
            "alpha_rgb": alpha_rgb,
            "alpha_t": alpha_t,
            "alpha_shared": alpha_shared,
        }

        fused_feat = alpha_rgb * rgb_e + alpha_t * thermal_e + alpha_shared * shared_e

        should_return_weights = (
            self.return_router_weights
            if return_router_weights is None
            else return_router_weights
        )
        if not should_return_weights:
            return fused_feat

        weights = {
            "alpha": alpha,
            "alpha_rgb": alpha_rgb,
            "alpha_t": alpha_t,
            "alpha_shared": alpha_shared,
        }
        return fused_feat, weights


class ClassAwareSemanticRouter(nn.Module):
    """
    Class-aware router with learnable class embeddings.

    Inputs:
        rgb_feat, thermal_feat, shared_feat: [B, C, H, W]

    Outputs:
        channel_wise=False: [B, num_classes, 3, 1, 1]
        channel_wise=True:  [B, num_classes, 3, C, 1, 1]
    """

    def __init__(
        self,
        num_classes: int,
        in_channels: int,
        embed_dim: int,
        channel_wise: bool = False,
        class_embedding_path: str = "",
        stage_name: str = "",
    ) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.in_channels = in_channels
        self.embed_dim = embed_dim
        self.channel_wise = channel_wise
        self.class_embedding_path = class_embedding_path
        self.stage_name = stage_name

        self.class_embeddings = nn.Embedding(num_classes, embed_dim)
        self.class_embedding_load_report: Dict[str, Any] = {
            "stage": self.stage_name,
            "status": "random_init",
            "success": False,
            "path": self.class_embedding_path,
            "expected_shape": [self.num_classes, self.embed_dim],
        }
        if self.class_embedding_path:
            loaded_tensor, load_report = _load_class_embedding_tensor(
                self.class_embedding_path,
                num_classes=self.num_classes,
                embed_dim=self.embed_dim,
            )
            with torch.no_grad():
                self.class_embeddings.weight.copy_(loaded_tensor)
            max_abs_diff = float(
                (self.class_embeddings.weight.detach().cpu() - loaded_tensor).abs().max().item()
            )
            load_report["stage"] = self.stage_name
            load_report["verified"] = max_abs_diff < 1e-7
            load_report["max_abs_diff"] = max_abs_diff
            self.class_embedding_load_report = load_report

        self.visual_pool = nn.AdaptiveAvgPool2d(1)
        self.visual_proj = nn.Sequential(
            nn.Conv2d(in_channels * 3, embed_dim, kernel_size=1, bias=True),
            nn.GELU(),
        )

        hidden_dim = max(16, embed_dim)
        routing_out_dim = 3 * in_channels if channel_wise else 3
        self.routing_mlp = nn.Sequential(
            nn.Linear(embed_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, routing_out_dim),
        )

    def forward(
        self,
        rgb_feat: torch.Tensor,
        thermal_feat: torch.Tensor,
        shared_feat: torch.Tensor,
        query_context: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if rgb_feat.dim() != 4 or thermal_feat.dim() != 4 or shared_feat.dim() != 4:
            raise ValueError("ClassAwareSemanticRouter expects 4D inputs [B, C, H, W].")
        if rgb_feat.shape != thermal_feat.shape or rgb_feat.shape != shared_feat.shape:
            raise ValueError("RGB, thermal, and shared features must have the same shape.")
        if rgb_feat.shape[1] != self.in_channels:
            raise ValueError(
                f"Expected feature channels {self.in_channels}, got {rgb_feat.shape[1]}."
            )

        visual_feat = torch.cat([rgb_feat, thermal_feat, shared_feat], dim=1)
        visual_context = self.visual_proj(self.visual_pool(visual_feat)).flatten(1)
        if query_context is not None:
            if query_context.dim() != 3:
                raise ValueError("query_context must have shape [B, num_classes, embed_dim].")
            if query_context.shape[1] != self.num_classes or query_context.shape[2] != self.embed_dim:
                raise ValueError(
                    f"Expected query_context shape [B, {self.num_classes}, {self.embed_dim}], "
                    f"got {tuple(query_context.shape)}."
                )
            visual_context = visual_context.unsqueeze(1).expand(-1, self.num_classes, -1)
            class_context = query_context + self.class_embeddings.weight.unsqueeze(0)
        else:
            visual_context = visual_context.unsqueeze(1).expand(-1, self.num_classes, -1)
            class_context = self.class_embeddings.weight.unsqueeze(0).expand(rgb_feat.shape[0], -1, -1)

        routing_input = torch.cat([visual_context, class_context], dim=-1)
        routing_logits = self.routing_mlp(routing_input)

        if self.channel_wise:
            routing_logits = routing_logits.view(
                rgb_feat.shape[0], self.num_classes, 3, self.in_channels, 1, 1
            )
            return torch.softmax(routing_logits, dim=2)

        routing_logits = routing_logits.view(rgb_feat.shape[0], self.num_classes, 3, 1, 1)
        return torch.softmax(routing_logits, dim=2)


class ClassAwareSeMoEFusionBlock(nn.Module):
    """
    Class-aware SeMoE fusion block.

    efficient_mode=False:
        returns fused_class_features with shape [B, num_classes, C, H, W]

    efficient_mode=True:
        avoids materializing the full class-specific feature tensor and returns:
            {
                "routing_weights": ...,
                "fused_feature": [B, C, H, W],
                "rgb_expert": [B, C, H, W],
                "thermal_expert": [B, C, H, W],
                "shared_expert": [B, C, H, W],
            }
    """

    def __init__(
        self,
        num_classes: int,
        in_channels: int,
        embed_dim: int,
        channel_wise: bool = False,
        efficient_mode: bool = False,
        expert_depth: int = 1,
        class_embedding_path: str = "",
        stage_name: str = "",
    ) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.in_channels = in_channels
        self.channel_wise = channel_wise
        self.efficient_mode = efficient_mode
        self.expert_depth = _validate_expert_depth(expert_depth)
        self.latest_router_weights = {}

        self.rgb_expert = RGBExpert(in_channels, depth=self.expert_depth) # RGB专家
        self.thermal_expert = ThermalExpert(in_channels, depth=self.expert_depth) # 红外专家
        self.shared_expert = SharedExpert(in_channels, depth=self.expert_depth) # 共享专家
        self.router = ClassAwareSemanticRouter(
            num_classes=num_classes,
            in_channels=in_channels,
            embed_dim=embed_dim,
            channel_wise=channel_wise,
            class_embedding_path=class_embedding_path,
            stage_name=stage_name,
        )
        self.class_embedding_load_report = dict(self.router.class_embedding_load_report)

    def forward(
        self,
        rgb_feat: torch.Tensor,
        thermal_feat: torch.Tensor,
    ) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
        if rgb_feat.dim() != 4 or thermal_feat.dim() != 4:
            raise ValueError("ClassAwareSeMoEFusionBlock expects 4D inputs [B, C, H, W].")
        if rgb_feat.shape != thermal_feat.shape:
            raise ValueError("RGB and thermal features must have the same shape.")
        if rgb_feat.shape[1] != self.in_channels:
            raise ValueError(
                f"Expected feature channels {self.in_channels}, got {rgb_feat.shape[1]}."
            )

        rgb_e = self.rgb_expert(rgb_feat)
        thermal_e = self.thermal_expert(thermal_feat)
        shared_e = self.shared_expert(rgb_feat, thermal_feat)
        routing_weights = self.router(rgb_e, thermal_e, shared_e)
        self.latest_router_weights = {
            "routing_weights": routing_weights,
        }

        if self.efficient_mode:
            if self.channel_wise:
                alpha_rgb = routing_weights[:, :, 0].mean(dim=1)
                alpha_t = routing_weights[:, :, 1].mean(dim=1)
                alpha_shared = routing_weights[:, :, 2].mean(dim=1)
            else:
                alpha_rgb = routing_weights[:, :, 0].mean(dim=1).unsqueeze(1)
                alpha_t = routing_weights[:, :, 1].mean(dim=1).unsqueeze(1)
                alpha_shared = routing_weights[:, :, 2].mean(dim=1).unsqueeze(1)

            fused_feature = alpha_rgb * rgb_e + alpha_t * thermal_e + alpha_shared * shared_e
            return {
                "routing_weights": routing_weights,
                "fused_feature": fused_feature,
                "rgb_expert": rgb_e,
                "thermal_expert": thermal_e,
                "shared_expert": shared_e,
            }

        rgb_expand = rgb_e.unsqueeze(1)
        thermal_expand = thermal_e.unsqueeze(1)
        shared_expand = shared_e.unsqueeze(1)

        if self.channel_wise:
            alpha_rgb = routing_weights[:, :, 0]
            alpha_t = routing_weights[:, :, 1]
            alpha_shared = routing_weights[:, :, 2]
        else:
            alpha_rgb = routing_weights[:, :, 0].unsqueeze(2)
            alpha_t = routing_weights[:, :, 1].unsqueeze(2)
            alpha_shared = routing_weights[:, :, 2].unsqueeze(2)

        fused_class_features = (
            alpha_rgb * rgb_expand
            + alpha_t * thermal_expand
            + alpha_shared * shared_expand
        )
        return fused_class_features


def _run_basic_shape_tests() -> None:
    """Simple smoke tests with random tensors."""

    device = torch.device("cpu")
    rgb_feat = torch.randn(2, 64, 32, 48, device=device)
    thermal_feat = torch.randn(2, 64, 32, 48, device=device)

    rgb_expert = RGBExpert(64).to(device)
    thermal_expert = ThermalExpert(64).to(device)
    shared_expert = SharedExpert(64).to(device)

    rgb_out = rgb_expert(rgb_feat)
    thermal_out = thermal_expert(thermal_feat)
    shared_out = shared_expert(rgb_feat, thermal_feat)

    assert rgb_out.shape == rgb_feat.shape
    assert thermal_out.shape == thermal_feat.shape
    assert shared_out.shape == rgb_feat.shape

    router_channel = SemanticExpertRouter(64, channel_wise=True).to(device)
    router_sample = SemanticExpertRouter(64, channel_wise=False).to(device)

    alpha_channel = router_channel(shared_out)
    alpha_sample = router_sample(shared_out)

    assert alpha_channel.shape == (2, 3, 64, 1, 1)
    assert alpha_sample.shape == (2, 3, 1, 1, 1)

    block_channel = SeMoEFusionBlock(
        channels=64,
        channel_wise_router=True,
        return_router_weights=True,
    ).to(device)
    fused_channel, weights_channel = block_channel(rgb_feat, thermal_feat)

    assert fused_channel.shape == rgb_feat.shape
    assert weights_channel["alpha"].shape == (2, 3, 64, 1, 1)

    block_sample = SeMoEFusionBlock(
        channels=64,
        channel_wise_router=False,
        return_router_weights=True,
    ).to(device)
    fused_sample, weights_sample = block_sample(rgb_feat, thermal_feat)

    assert fused_sample.shape == rgb_feat.shape
    assert weights_sample["alpha"].shape == (2, 3, 1, 1, 1)

    class_router_sample = ClassAwareSemanticRouter(
        num_classes=9,
        in_channels=128,
        embed_dim=64,
        channel_wise=False,
    ).to(device)
    class_router_channel = ClassAwareSemanticRouter(
        num_classes=9,
        in_channels=128,
        embed_dim=64,
        channel_wise=True,
    ).to(device)

    rgb_feat_large = torch.randn(2, 128, 64, 64, device=device)
    thermal_feat_large = torch.randn(2, 128, 64, 64, device=device)
    shared_feat_large = torch.randn(2, 128, 64, 64, device=device)

    class_alpha_sample = class_router_sample(rgb_feat_large, thermal_feat_large, shared_feat_large)
    class_alpha_channel = class_router_channel(rgb_feat_large, thermal_feat_large, shared_feat_large)

    assert class_alpha_sample.shape == (2, 9, 3, 1, 1)
    assert class_alpha_channel.shape == (2, 9, 3, 128, 1, 1)

    class_block = ClassAwareSeMoEFusionBlock(
        num_classes=9,
        in_channels=128,
        embed_dim=64,
        channel_wise=False,
        efficient_mode=False,
    ).to(device)
    fused_class_features = class_block(rgb_feat_large, thermal_feat_large)
    assert fused_class_features.shape == (2, 9, 128, 64, 64)

    class_block_channel = ClassAwareSeMoEFusionBlock(
        num_classes=9,
        in_channels=128,
        embed_dim=64,
        channel_wise=True,
        efficient_mode=False,
    ).to(device)
    fused_class_features_channel = class_block_channel(rgb_feat_large, thermal_feat_large)
    assert fused_class_features_channel.shape == (2, 9, 128, 64, 64)

    class_block_efficient = ClassAwareSeMoEFusionBlock(
        num_classes=9,
        in_channels=128,
        embed_dim=64,
        channel_wise=False,
        efficient_mode=True,
    ).to(device)
    efficient_outputs = class_block_efficient(rgb_feat_large, thermal_feat_large)
    assert efficient_outputs["routing_weights"].shape == (2, 9, 3, 1, 1)
    assert efficient_outputs["fused_feature"].shape == (2, 128, 64, 64)

    print("SeMoE-Fusion shape tests passed.")


if __name__ == "__main__":
    _run_basic_shape_tests()
