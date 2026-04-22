import torch
from torch import nn
from torch.nn import functional as F

from detectron2.config import configurable
from detectron2.layers import Conv2d

from models.modules.semoe_fusion import (
    RGBExpert,
    ThermalExpert,
    SharedExpert,
    ClassAwareSemanticRouter,
)
from .maskformer_transformer_decoder import TRANSFORMER_DECODER_REGISTRY


class DecoderFFN(nn.Module):
    def __init__(self, dim, hidden_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, dim),
        )
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        return self.norm(x + self.net(x))


class MaskMLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        dims = [input_dim] + [hidden_dim] * (num_layers - 1) + [output_dim]
        self.layers = nn.ModuleList(
            nn.Linear(dims[i], dims[i + 1]) for i in range(num_layers)
        )

    def forward(self, x):
        for idx, layer in enumerate(self.layers):
            x = layer(x)
            if idx < self.num_layers - 1:
                x = F.gelu(x)
        return x


class SemanticQueryDecoderLayer(nn.Module):
    def __init__(self, decoder_dim, num_heads):
        super().__init__()
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=decoder_dim,
            num_heads=num_heads,
            dropout=0.0,
            batch_first=True,
        )
        self.norm1 = nn.LayerNorm(decoder_dim)
        self.ffn = DecoderFFN(decoder_dim, decoder_dim * 4)

    def forward(self, queries, visual_tokens):
        attn_out, _ = self.cross_attn(queries, visual_tokens, visual_tokens)
        queries = self.norm1(queries + attn_out)
        queries = self.ffn(queries)
        return queries


@TRANSFORMER_DECODER_REGISTRY.register()
class SemanticQueryDecoder(nn.Module):
    @configurable
    def __init__(
        self,
        in_channels,
        mask_classification=True,
        *,
        num_classes: int,
        in_channels_list,
        decoder_dim: int,
        num_layers: int,
        num_heads: int,
        recursive_rerouting: bool,
        max_recursive_layers: int,
        recursive_channel_wise: bool,
    ):
        super().__init__()
        self.mask_classification = mask_classification
        self.num_classes = num_classes
        self.decoder_dim = decoder_dim
        self.recursive_rerouting = recursive_rerouting
        self.max_recursive_layers = max(0, min(max_recursive_layers, num_layers))
        self.recursive_channel_wise = recursive_channel_wise

        if isinstance(in_channels_list, int):
            in_channels_list = [in_channels_list]
        self.in_channels_list = list(in_channels_list)

        self.input_proj = nn.ModuleList(
            [Conv2d(ch, decoder_dim, kernel_size=1) for ch in self.in_channels_list]
        )
        self.class_queries = nn.Embedding(num_classes, decoder_dim)
        self.decoder_layers = nn.ModuleList(
            [SemanticQueryDecoderLayer(decoder_dim, num_heads) for _ in range(num_layers)]
        )

        self.mask_embed = MaskMLP(decoder_dim, decoder_dim, decoder_dim, 3)
        self.pixel_embed = Conv2d(decoder_dim, decoder_dim, kernel_size=1)
        self.class_embed = nn.Linear(decoder_dim, num_classes + 1)
        self.latest_aux_outputs = []

        # Recursive expert-aware decoding is optional and disabled by default.
        self.recursive_rgb_expert = RGBExpert(decoder_dim)
        self.recursive_thermal_expert = ThermalExpert(decoder_dim)
        self.recursive_shared_expert = SharedExpert(decoder_dim)
        self.recursive_router = ClassAwareSemanticRouter(
            num_classes=num_classes,
            in_channels=decoder_dim,
            embed_dim=decoder_dim,
            channel_wise=recursive_channel_wise,
        )

    @classmethod
    def from_config(cls, cfg, in_channels, mask_classification):
        if isinstance(in_channels, (list, tuple)):
            in_channels_list = list(in_channels)
        else:
            in_channels_list = [in_channels]

        return {
            "in_channels": in_channels,
            "mask_classification": mask_classification,
            "num_classes": cfg.MODEL.SEM_SEG_HEAD.NUM_CLASSES,
            "in_channels_list": in_channels_list,
            "decoder_dim": cfg.MODEL.MASK_FORMER.HIDDEN_DIM,
            "num_layers": max(1, cfg.MODEL.MASK_FORMER.DEC_LAYERS - 1),
            "num_heads": cfg.MODEL.MASK_FORMER.NHEADS,
            "recursive_rerouting": cfg.MODEL.MASK_FORMER.RECURSIVE_REROUTING,
            "max_recursive_layers": cfg.MODEL.MASK_FORMER.MAX_RECURSIVE_LAYERS,
            "recursive_channel_wise": cfg.MODEL.MASK_FORMER.RECURSIVE_CHANNEL_WISE,
        }

    def _normalize_inputs(self, features):
        if isinstance(features, torch.Tensor):
            return [features]
        if isinstance(features, (list, tuple)):
            return list(features)
        raise TypeError("SemanticQueryDecoder expects a tensor or a list/tuple of tensors.")

    def _project_feature(self, feature, proj_idx):
        if len(self.input_proj) == 1:
            return self.input_proj[0](feature)
        for in_channels, proj in zip(self.in_channels_list, self.input_proj):
            if feature.shape[1] == in_channels:
                return proj(feature)
        return self.input_proj[proj_idx](feature)

    def _build_visual_tokens(self, projected_features):
        return torch.cat(
            [feat.flatten(2).transpose(1, 2) for feat in projected_features],
            dim=1,
        )

    def _compute_logits(self, queries, pixel_feature):
        mask_embed = self.mask_embed(queries)
        pixel_embed = self.pixel_embed(pixel_feature)
        sem_seg_logits = torch.einsum("bkd,bdhw->bkhw", mask_embed, pixel_embed)
        pred_logits = self.class_embed(queries)
        return pred_logits, sem_seg_logits

    def _recursive_reroute(self, pixel_feature, queries):
        rgb_e = self.recursive_rgb_expert(pixel_feature)
        thermal_e = self.recursive_thermal_expert(pixel_feature)
        shared_e = self.recursive_shared_expert(pixel_feature, pixel_feature)
        routing_weights = self.recursive_router(
            rgb_e,
            thermal_e,
            shared_e,
            query_context=queries,
        )

        if self.recursive_channel_wise:
            alpha_rgb = routing_weights[:, :, 0].mean(dim=1)
            alpha_t = routing_weights[:, :, 1].mean(dim=1)
            alpha_shared = routing_weights[:, :, 2].mean(dim=1)
        else:
            alpha_rgb = routing_weights[:, :, 0].mean(dim=1).unsqueeze(1)
            alpha_t = routing_weights[:, :, 1].mean(dim=1).unsqueeze(1)
            alpha_shared = routing_weights[:, :, 2].mean(dim=1).unsqueeze(1)

        fused_feature = alpha_rgb * rgb_e + alpha_t * thermal_e + alpha_shared * shared_e
        return fused_feature, routing_weights

    def forward(self, x, mask_features=None, mask=None):
        del mask_features
        del mask

        features = self._normalize_inputs(x)
        projected_features = []
        pixel_feature = None
        pixel_area = -1

        for idx, feature in enumerate(features):
            proj_feature = self._project_feature(feature, min(idx, len(self.input_proj) - 1))
            projected_features.append(proj_feature)
            area = proj_feature.shape[-2] * proj_feature.shape[-1]
            if area > pixel_area:
                pixel_area = area
                pixel_feature = proj_feature

        visual_tokens = self._build_visual_tokens(projected_features)
        batch_size = visual_tokens.shape[0]

        queries = self.class_queries.weight.unsqueeze(0).expand(batch_size, -1, -1)
        aux_outputs = []
        recursive_start = len(self.decoder_layers) - self.max_recursive_layers
        for layer_idx, layer in enumerate(self.decoder_layers):
            queries = layer(queries, visual_tokens)
            pred_logits, sem_seg_logits = self._compute_logits(queries, pixel_feature)

            aux_entry = {
                "pred_logits": pred_logits,
                "pred_masks": sem_seg_logits,
                "sem_seg_logits": sem_seg_logits,
            }

            if self.recursive_rerouting and layer_idx >= recursive_start:
                pixel_feature, router_weights = self._recursive_reroute(pixel_feature, queries)
                projected_features[0] = pixel_feature
                visual_tokens = self._build_visual_tokens(projected_features)
                aux_entry["router_weights"] = router_weights

            aux_outputs.append(aux_entry)
        pred_logits, sem_seg_logits = self._compute_logits(queries, pixel_feature)
        self.latest_aux_outputs = aux_outputs

        return {
            "pred_logits": pred_logits,
            "pred_masks": sem_seg_logits,
            "aux_outputs": aux_outputs[:-1],
            "sem_seg_logits": sem_seg_logits,
        }


if __name__ == "__main__":
    decoder = SemanticQueryDecoder(
        in_channels=256,
        mask_classification=True,
        num_classes=9,
        in_channels_list=[128, 256, 256, 256],
        decoder_dim=256,
        num_layers=3,
        num_heads=8,
        recursive_rerouting=False,
        max_recursive_layers=2,
        recursive_channel_wise=False,
    )
    multi_scale_features = [
        torch.randn(2, 128, 64, 64),
        torch.randn(2, 256, 32, 32),
        torch.randn(2, 256, 16, 16),
        torch.randn(2, 256, 8, 8),
    ]
    out = decoder(multi_scale_features)
    print("multi_scale logits:", tuple(out["sem_seg_logits"].shape))
    print("multi_scale recursive off aux:", len(out["aux_outputs"]))

    single_feature = torch.randn(2, 256, 32, 32)
    out_single = decoder(single_feature)
    print("single_scale logits:", tuple(out_single["sem_seg_logits"].shape))

    recursive_decoder = SemanticQueryDecoder(
        in_channels=256,
        mask_classification=True,
        num_classes=9,
        in_channels_list=[128, 256, 256, 256],
        decoder_dim=256,
        num_layers=3,
        num_heads=8,
        recursive_rerouting=True,
        max_recursive_layers=2,
        recursive_channel_wise=True,
    )
    out_recursive = recursive_decoder(multi_scale_features)
    print("multi_scale recursive on logits:", tuple(out_recursive["sem_seg_logits"].shape))
    print("multi_scale recursive on aux:", len(out_recursive["aux_outputs"]))
