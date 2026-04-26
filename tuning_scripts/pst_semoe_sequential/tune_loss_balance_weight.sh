#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"
DEFAULT_DATASET_DIR="/media/wislab/Datasets_SSD2T1/lzh/datasets/PST900_RGBT_Dataset"
if [[ ! -d "$DEFAULT_DATASET_DIR" ]]; then
  DEFAULT_DATASET_DIR="/home/wislab/lzh/datasets/PST900_RGBT_Dataset"
fi
DATASET_DIR="${DATASET_DIR:-$DEFAULT_DATASET_DIR}"
PRETRAINED_PATH="${PRETRAINED_PATH:-$PROJECT_ROOT/pretrained_model/swinv2_tiny_patch4_window16_256.pth}"
CONFIG_BASE="${CONFIG_BASE:-$PROJECT_ROOT/configs/PSTdataset/ablation/semoe_class_router.yaml}"
WORK_DIR="${WORK_DIR:-$PROJECT_ROOT/checkpoints/tune_loss_balance_weight}"
NUM_GPUS="${NUM_GPUS:-1}"
IMS_PER_BATCH="${IMS_PER_BATCH:-8}"
CHECK_VAL_EVERY_N_EPOCH="${CHECK_VAL_EVERY_N_EPOCH:-1}"
CLASS_EMBED_DIM="${CLASS_EMBED_DIM:-256}"
EXPERT_DEPTH="${EXPERT_DEPTH:-1}"
CLASS_INDEPENDENT="${CLASS_INDEPENDENT:-false}"
USE_CLASS_PROBE_LOSS="${USE_CLASS_PROBE_LOSS:-false}"
LOSS_CLASS_PROBE_WEIGHT="${LOSS_CLASS_PROBE_WEIGHT:-0.2}"
EPOCHS="${EPOCHS:-250}"
# Fine sweep around the current best region.
# Override with: VALUES="0.0 0.0005 0.001 0.002 0.003" bash ...
VALUES_STRING="${VALUES:-0.0005 0.001 0.002 0.003}"
read -r -a VALUES <<< "$VALUES_STRING"

mkdir -p "$WORK_DIR"

if [[ ! -d "$DATASET_DIR/train/rgb" ]]; then
  echo "Missing dataset directory: $DATASET_DIR/train/rgb"
  exit 1
fi

TRAIN_SAMPLES=$(find -L "$DATASET_DIR/train/rgb" \( -type f -o -type l \) | wc -l | tr -d ' ')
if [[ "$TRAIN_SAMPLES" -lt "$IMS_PER_BATCH" ]]; then
  echo "Invalid train sample count: $TRAIN_SAMPLES"
  echo "Resolved dataset dir: $DATASET_DIR"
  echo "Example files under train/rgb:"
  find "$DATASET_DIR/train/rgb" -maxdepth 1 \( -type f -o -type l \) | sed -n '1,10p'
  exit 1
fi

STEPS_PER_EPOCH=$(( TRAIN_SAMPLES / IMS_PER_BATCH ))
MAX_ITER=$(( STEPS_PER_EPOCH * EPOCHS ))
RUN_TAG="$(date +%Y%m%d_%H%M%S)"
SUMMARY_FILE="$WORK_DIR/sweep_summary_${RUN_TAG}.txt"
LATEST_SUMMARY_LINK="$WORK_DIR/sweep_summary_latest.txt"

{
  echo "parameter: LOSS_BALANCE_WEIGHT"
  echo "values: ${VALUES[*]}"
  echo "epochs: $EPOCHS"
  echo "train_samples: $TRAIN_SAMPLES"
  echo "ims_per_batch: $IMS_PER_BATCH"
  echo "steps_per_epoch: $STEPS_PER_EPOCH"
  echo "max_iter: $MAX_ITER"
  echo
} > "$SUMMARY_FILE"
ln -sfn "$SUMMARY_FILE" "$LATEST_SUMMARY_LINK"

for VALUE in "${VALUES[@]}"; do
  SAFE_VALUE="${VALUE//./p}"
  EXP_NAME="pst_lb_fine_${SAFE_VALUE}_ep${EPOCHS}"
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
    EXPERT_DEPTH: $EXPERT_DEPTH
    CLASS_INDEPENDENT: $CLASS_INDEPENDENT
  MASK_FORMER:
    DECODER_TYPE: "baseline"
    RECURSIVE_REROUTING: False
    LOSS_BALANCE_WEIGHT: $VALUE
    USE_CLASS_PROBE_LOSS: $USE_CLASS_PROBE_LOSS
    LOSS_CLASS_PROBE_WEIGHT: $LOSS_CLASS_PROBE_WEIGHT
    LOSS_AUX_WEIGHT: 0.0
    USE_CONSISTENCY_LOSS: False
    LOSS_CONSISTENCY_WEIGHT: 0.0
EOF

  echo "==================================================" | tee -a "$SUMMARY_FILE"
  echo "start: $(date '+%F %T')" | tee -a "$SUMMARY_FILE"
  echo "exp_name: $EXP_NAME" | tee -a "$SUMMARY_FILE"
  echo "LOSS_BALANCE_WEIGHT: $VALUE" | tee -a "$SUMMARY_FILE"
  echo "CLASS_EMBED_DIM: $CLASS_EMBED_DIM" | tee -a "$SUMMARY_FILE"
  echo "EXPERT_DEPTH: $EXPERT_DEPTH" | tee -a "$SUMMARY_FILE"
  echo "CLASS_INDEPENDENT: $CLASS_INDEPENDENT" | tee -a "$SUMMARY_FILE"
  echo "USE_CLASS_PROBE_LOSS: $USE_CLASS_PROBE_LOSS" | tee -a "$SUMMARY_FILE"
  echo "LOSS_CLASS_PROBE_WEIGHT: $LOSS_CLASS_PROBE_WEIGHT" | tee -a "$SUMMARY_FILE"
  echo "config: $TMP_CONFIG" | tee -a "$SUMMARY_FILE"

  "$PYTHON_BIN" "$PROJECT_ROOT/train.py" \
    --config-file "$TMP_CONFIG" \
    --name "$EXP_NAME" \
    --work_dir "$WORK_DIR" \
    --num-gpus "$NUM_GPUS" \
    --seed 1024 \
    --check_val_every_n_epoch "$CHECK_VAL_EVERY_N_EPOCH"

  LOG_FILE="$WORK_DIR/$EXP_NAME/log.txt"
  BEST_LINE="$(grep 'best_mIoU' "$LOG_FILE" | tail -n 1 || true)"
  if [[ -z "$BEST_LINE" ]]; then
    BEST_LINE="best_mIoU: not_found"
  fi
  echo "$BEST_LINE" | tee -a "$SUMMARY_FILE"
  echo "finish: $(date '+%F %T')" | tee -a "$SUMMARY_FILE"
  echo | tee -a "$SUMMARY_FILE"
done
