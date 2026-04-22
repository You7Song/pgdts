#!/bin/bash
# =============================================================================
# Stage 3 Training Script: Decision Generation
# =============================================================================
# Final stage: generates concise, actionable avoidance instructions for
# visually impaired users based on structured perception data.
#
# Loss: L_Stage3 = L_AR^decision  (standard autoregressive SFT)
#
# Usage:
#   bash scripts/train_stage3.sh
# =============================================================================

set -e

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "${PROJECT_ROOT}"

echo "[INFO] Stage 3 — Decision Generation (Standard SFT)"

STAGE2_ADAPTER="./checkpoints/stage2_lora"

# ---------------------------------------------------------------------------
# Data paths — adjust these to point to your actual dataset files.
# ---------------------------------------------------------------------------
TRAIN_JSONL="./data/train.jsonl"
VAL_JSONL="./data/val.jsonl"

python3 src/train_stage3.py \
  --train_jsonl ${TRAIN_JSONL} \
  --val_jsonl ${VAL_JSONL} \
  --output_dir ./checkpoints/stage3_lora \
  --adapter_path ${STAGE2_ADAPTER} \
  --num_train_epochs 2 \
  --learning_rate 1e-4 \
  --lora_rank 24 \
  --lora_alpha 48 \
  --max_length 2048 \
  --per_device_train_batch_size 4 \
  --gradient_accumulation_steps 4 \
  --save_steps 100 \
  --eval_steps 100 \
  --logging_steps 10

echo "[INFO] Stage 3 complete. Weights saved to ./checkpoints/stage3_lora"
