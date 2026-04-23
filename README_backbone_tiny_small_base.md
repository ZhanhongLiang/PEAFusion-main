# Tiny / Small / Base Backbone 配置说明（PST + SeMoE）

这份说明用于统一三档 backbone（`tiny/small/base`）在你当前 SeMoE 方案下的配置，重点覆盖：
- 模型规模（按 stage 宽度/深度定义）
- 专家配置（SeMoE）
- `class_embed_dim` 配置
- decoder 配置

## 1. 统一前提（建议固定不变）

- Fusion: `USE_SEMOE_FUSION=True`
- Fusion type: `FUSION_TYPE="semoe"`
- Router: `ROUTER_TYPE="class_aware"`
- Channel-wise routing: `SEMOE_CHANNEL_WISE=True`
- Decoder: `DECODER_TYPE="baseline"`
- Recursive rerouting: `RECURSIVE_REROUTING=False`
- Mask2Former decoder layers: `DEC_LAYERS=4`

以上与当前实验主线一致，便于公平对比。

---

## 2. 三档 backbone 推荐表

| 档位 | 来源 | SWIN.EMBED_DIM | SWIN.DEPTHS | SWIN.NUM_HEADS | Stage 通道（res2/3/4/5） | class_embed_dim | 说明 |
|---|---|---:|---|---|---|---:|---|
| tiny | 沿用 PEAFusion | 96 | [2,2,6,2] | [3,6,12,24] | [96,192,384,768] | 256 | 直接用现有 `swin_v2_tiny.yaml` |
| small | 推荐新增 | 96 | [2,2,18,2] | [3,6,12,24] | [96,192,384,768] | 256 | 仓库无现成 small，建议按 SwinV2-S 宽度/深度 |
| base | 沿用 PEAFusion + 你当前建议 | 128 | [2,2,18,2] | [4,8,16,32] | [128,256,512,1024] | 384 | 直接用现有 `swin_v2_base.yaml`，class_embed 提到 384 |

说明：
- `Stage 通道`来自实现：`num_features = embed_dim * 2^i`。
- `small` 与 `tiny`通道一致，但第3 stage 深度从 6 增到 18，因此容量明显提升。

---

## 3. 专家配置（SeMoE）在三档下如何变化

你的实现中，每个 stage 1 个 fusion block，共 4 个 stage：
- stage 对应：`res2, res3, res4, res5`
- 每个 block 固定 3 个专家：`RGBExpert / ThermalExpert / SharedExpert`
- 因此总是 4 组 block × 3 专家（结构不变，参数量随 stage 通道变化）

即：
- `tiny/small`: 专家通道为 `[96,192,384,768]`
- `base`: 专家通道为 `[128,256,512,1024]`

---

## 4. Decoder 配置（建议三档一致）

为了让 backbone 对比干净，decoder 建议固定：
- `DECODER_TYPE="baseline"`
- `TRANSFORMER_DECODER_NAME="MultiScaleMaskedTransformerDecoder"`
- `DEC_LAYERS=4`
- `RECURSIVE_REROUTING=False`
- `LOSS_AUX_WEIGHT=0.0`（若你当前主线已这样设置）
- `USE_CONSISTENCY_LOSS=False`

---

## 5. 可直接落地的配置映射

### tiny
- 配置基线：`configs/PSTdataset/swin_v2/swin_v2_tiny.yaml`
- 覆盖项：
  - `MODEL.FUSION.USE_SEMOE_FUSION=True`
  - `MODEL.FUSION.FUSION_TYPE="semoe"`
  - `MODEL.FUSION.ROUTER_TYPE="class_aware"`
  - `MODEL.FUSION.CLASS_EMBED_DIM=256`
  - `MODEL.MASK_FORMER.DECODER_TYPE="baseline"`
  - `MODEL.MASK_FORMER.RECURSIVE_REROUTING=False`

### small（新增）
- 参考 tiny 配置，改为：
  - `MODEL.SWIN.DEPTHS=[2,2,18,2]`
  - `MODEL.SWIN.NUM_HEADS=[3,6,12,24]`
  - `MODEL.SWIN.EMBED_DIM=96`
  - `MODEL.FUSION.CLASS_EMBED_DIM=256`
- 预训练权重需与 `WINDOW_SIZE/PRETRAINED_WINDOW_SIZE` 匹配。
  - 如果你拿到的是 window16 预训练，就保持 `WINDOW_SIZE=16`。
  - 如果是 window12 预训练，就同步改为 12。

### base
- 配置基线：`configs/PSTdataset/swin_v2/swin_v2_base.yaml`
- 覆盖项同 tiny，但：
  - `MODEL.FUSION.CLASS_EMBED_DIM=384`

---

## 6. 结论（你当前可直接采用）

- `tiny/small` 用 `class_embed_dim=256`，`base` 用 `class_embed_dim=384` 是合理设置。
- 这套方案满足“尽量沿用 PEAFusion；无现成 small 时给出可解释的合理配置”。
- 论文里可写：除 backbone 宽深和 class embedding 维度分级设置外，fusion/decoder/训练策略保持一致。

