#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"

# Dataset root expected by the PST dataloader:
#   train/rgb, train/thermal, train/labels
#   test/rgb,  test/thermal,  test/labels
DATASET_DIR="${DATASET_DIR:-/home/wislab/lzh/datasets/PST900_RGBT_Dataset}"

# Official SwinV2 Tiny pretrained checkpoint.
BACKBONE_SIZE="${BACKBONE_SIZE:-tiny}"
case "$BACKBONE_SIZE" in
  tiny)
    DEFAULT_CONFIG_BASE="$PROJECT_ROOT/configs/PSTdataset/swin_v2/swin_v2_tiny.yaml"
    DEFAULT_PRETRAINED_PATH="$PROJECT_ROOT/pretrained_model/swinv2_tiny_patch4_window16_256.pth"
    ;;
  small)
    DEFAULT_CONFIG_BASE="$PROJECT_ROOT/configs/PSTdataset/swin_v2/swin_v2_small.yaml"
    DEFAULT_PRETRAINED_PATH="$PROJECT_ROOT/pretrained_model/swinv2_small_patch4_window16_256.pth"
    ;;
  base)
    DEFAULT_CONFIG_BASE="$PROJECT_ROOT/configs/PSTdataset/swin_v2/swin_v2_base.yaml"
    DEFAULT_PRETRAINED_PATH="$PROJECT_ROOT/pretrained_model/swinv2_base_patch4_window12_192_22k.pth"
    ;;
  *)
    echo "Unsupported BACKBONE_SIZE: $BACKBONE_SIZE"
    echo "Expected one of: tiny, small, base"
    exit 1
    ;;
esac

PRETRAINED_PATH="${PRETRAINED_PATH:-$DEFAULT_PRETRAINED_PATH}"
CONFIG_BASE="${CONFIG_BASE:-$DEFAULT_CONFIG_BASE}"
NUM_GPUS="${NUM_GPUS:-1}"
IMS_PER_BATCH="${IMS_PER_BATCH:-2}" # for debug 的batch_size设置为2
CLASS_EMBED_DIM="${CLASS_EMBED_DIM:-768}"
DEFAULT_CLASS_EMBEDDING_PATH="$PROJECT_ROOT/cls_embed/pst_class_embedding.pt"
CLASS_EMBEDDING_PATH="${CLASS_EMBEDDING_PATH:-$DEFAULT_CLASS_EMBEDDING_PATH}"
EXPERT_DEPTH="${EXPERT_DEPTH:-1}"
CLASS_INDEPENDENT="${CLASS_INDEPENDENT:-false}"
USE_CLASS_PROBE_LOSS="${USE_CLASS_PROBE_LOSS:-false}"
LOSS_CLASS_PROBE_WEIGHT="${LOSS_CLASS_PROBE_WEIGHT:-0.2}"
SEED="${SEED:-1024}"
WORK_DIR="${WORK_DIR:-$PROJECT_ROOT/checkpoints_debug}"
EXP_NAME="${EXP_NAME:-semoe_class_router_pst_debug}"
CHECK_VAL_EVERY_N_EPOCH="${CHECK_VAL_EVERY_N_EPOCH:-1}"
EPOCHS="${EPOCHS:-20}"
RESUME_CKPT_PATH="${RESUME_CKPT_PATH:-}"

if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
  echo "Python binary not found: $PYTHON_BIN"
  exit 1
fi

if [[ ! -f "$CONFIG_BASE" ]]; then
  echo "Base config not found: $CONFIG_BASE"
  exit 1
fi

if [[ ! -d "$DATASET_DIR/train/rgb" ]] || [[ ! -d "$DATASET_DIR/train/thermal" ]] || [[ ! -d "$DATASET_DIR/train/labels" ]]; then
  echo "Dataset layout does not match PST dataloader expectation:"
  echo "Expected directories:"
  echo "  $DATASET_DIR/train/rgb"
  echo "  $DATASET_DIR/train/thermal"
  echo "  $DATASET_DIR/train/labels"
  echo "  $DATASET_DIR/test/rgb"
  echo "  $DATASET_DIR/test/thermal"
  echo "  $DATASET_DIR/test/labels"
  exit 1
fi

if [[ ! -f "$PRETRAINED_PATH" ]]; then
  echo "Required pretrained checkpoint not found:"
  echo "  $PRETRAINED_PATH"
  echo
  echo "Please download:"
  echo "  SwinV2 Tiny patch4 window16 256 pretrained checkpoint"
  echo "  Recommended filename: swinv2_tiny_patch4_window16_256.pth"
  exit 1
fi

if [[ -n "$CLASS_EMBEDDING_PATH" ]] && [[ ! -f "$CLASS_EMBEDDING_PATH" ]]; then
  echo "Class embedding file not found:"
  echo "  $CLASS_EMBEDDING_PATH"
  exit 1
fi

TRAIN_SAMPLES="$(find -L "$DATASET_DIR/train/rgb" -maxdepth 1 -type f | wc -l | tr -d ' ')"
DEBUG_BATCH_SIZE=$IMS_PER_BATCH
if (( DEBUG_BATCH_SIZE > 2 )); then
  DEBUG_BATCH_SIZE=2
fi
EFFECTIVE_GLOBAL_BATCH_SIZE=$(( DEBUG_BATCH_SIZE * NUM_GPUS ))
if (( EFFECTIVE_GLOBAL_BATCH_SIZE <= 0 )); then
  echo "Invalid effective batch size: $EFFECTIVE_GLOBAL_BATCH_SIZE"
  exit 1
fi
STEPS_PER_EPOCH=$(( TRAIN_SAMPLES / EFFECTIVE_GLOBAL_BATCH_SIZE ))
if (( STEPS_PER_EPOCH < 1 )); then
  echo "Train sample count ($TRAIN_SAMPLES) is smaller than effective batch size ($EFFECTIVE_GLOBAL_BATCH_SIZE)."
  echo "Please reduce IMS_PER_BATCH or NUM_GPUS."
  exit 1
fi
MAX_ITER=$(( EPOCHS * STEPS_PER_EPOCH ))

TMP_CONFIG="/tmp/${EXP_NAME}_$(date +%Y%m%d_%H%M%S).yaml"

cat > "$TMP_CONFIG" <<EOF
_BASE_: $CONFIG_BASE

DATASETS:
  DIR: "$DATASET_DIR"

SOLVER:
  IMS_PER_BATCH: $IMS_PER_BATCH
  MAX_ITER: $MAX_ITER

MODEL:
  SWIN:
    PRETRAINED: "$PRETRAINED_PATH"
    MODEL_OUTPUT_ATTN: False
  FUSION:
    USE_SEMOE_FUSION: True
    FUSION_TYPE: "semoe"
    ROUTER_TYPE: "class_aware"
    SEMOE_CHANNEL_WISE: True
    CLASS_EMBED_DIM: $CLASS_EMBED_DIM
    CLASS_EMBEDDING_PATH: "$CLASS_EMBEDDING_PATH"
    EXPERT_DEPTH: $EXPERT_DEPTH
    CLASS_INDEPENDENT: $CLASS_INDEPENDENT
  MASK_FORMER:
    DECODER_TYPE: "baseline"
    RECURSIVE_REROUTING: False
    LOSS_BALANCE_WEIGHT: 0.01
    USE_CLASS_PROBE_LOSS: $USE_CLASS_PROBE_LOSS
    LOSS_CLASS_PROBE_WEIGHT: $LOSS_CLASS_PROBE_WEIGHT
    LOSS_AUX_WEIGHT: 0.0
    USE_CONSISTENCY_LOSS: False
    LOSS_CONSISTENCY_WEIGHT: 0.0
EOF

echo "Project root: $PROJECT_ROOT"
echo "Python: $PYTHON_BIN"
echo "Dataset: $DATASET_DIR"
echo "Backbone size: $BACKBONE_SIZE"
echo "Pretrained: $PRETRAINED_PATH"
echo "Config: $TMP_CONFIG"
echo "Experiment: $EXP_NAME"
echo "Work dir: $WORK_DIR"
echo
echo "Core switches:"
echo "  ims_per_batch=$IMS_PER_BATCH"
echo "  debug_batch_size=$DEBUG_BATCH_SIZE"
echo "  epochs=$EPOCHS"
echo "  train_samples=$TRAIN_SAMPLES"
echo "  effective_global_batch_size=$EFFECTIVE_GLOBAL_BATCH_SIZE"
echo "  steps_per_epoch=$STEPS_PER_EPOCH"
echo "  max_iter=$MAX_ITER"
echo "  backbone_size=$BACKBONE_SIZE"
echo "  class_embed_dim=$CLASS_EMBED_DIM"
echo "  class_embedding_path=${CLASS_EMBEDDING_PATH:-<random_init>}"
echo "  expert_depth=$EXPERT_DEPTH"
echo "  class_independent=$CLASS_INDEPENDENT"
echo "  use_class_probe_loss=$USE_CLASS_PROBE_LOSS"
echo "  loss_class_probe_weight=$LOSS_CLASS_PROBE_WEIGHT"
echo "  use_semoe_fusion=True"
echo "  fusion_type=semoe"
echo "  router_type=class_aware"
echo "  decoder_type=baseline"
echo "  recursive_rerouting=False"
echo "  loss_balance_weight=0.01"
echo "  loss_aux_weight=0.0"
echo "  consistency_loss=off"
echo

CMD=(
  "$PYTHON_BIN" "$PROJECT_ROOT/train_debug.py"
  --config-file "$TMP_CONFIG"
  --name "$EXP_NAME"
  --work_dir "$WORK_DIR"
  --num-gpus "$NUM_GPUS"
  --seed "$SEED"
  --check_val_every_n_epoch "$CHECK_VAL_EVERY_N_EPOCH"
)

if [[ -n "$RESUME_CKPT_PATH" ]]; then
  CMD+=(--resume_ckpt_path "$RESUME_CKPT_PATH")
fi

echo "Running:"
printf ' %q' "${CMD[@]}"
echo

"${CMD[@]}"
