#!/bin/bash
# =============================================================================
# Stage 2 Training Script: Spatial Understanding
# =============================================================================
# Trains the model to predict clock-face direction and absolute distance
# (in steps) for each detected object, using a token-level Distance
# Regression Head with distance-aware Huber loss.
#
# Loss: L_Stage2 = L_AR + λ_dist · L_dist   (λ_dist=0.2, δ=1.0, τ=5)
#
# Data format (JSONL):
#   {
#     "messages": [...],
#     "images": [...],
#     "spatial_info": [                          # optional; auto-parsed if absent
#       {"object": "vehicle", "distance_steps": 2.0, "distance_token_text": "2"}
#     ]
#   }
#
# Usage:
#   bash scripts/train_stage2.sh
# =============================================================================

set -e

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "${PROJECT_ROOT}"

echo "[INFO] Stage 2 — Spatial Understanding (Distance Regression Loss)"

STAGE1_ADAPTER="./checkpoints/stage1_lora"
if [ ! -d "${STAGE1_ADAPTER}" ]; then
    echo "[WARN] Stage 1 adapter not found at ${STAGE1_ADAPTER}. Training from base model."
fi

python3 src/train_stage2.py \
  --train_jsonl ./data/stage2_train.jsonl \
  --val_jsonl ./data/stage2_val.jsonl \
  --output_dir ./checkpoints/stage2_lora \
  --adapter_path ${STAGE1_ADAPTER} \
  --num_train_epochs 4 \
  --learning_rate 1.5e-4 \
  --lora_rank 24 \
  --lora_alpha 48 \
  --lambda_dist 0.2 \
  --huber_delta 1.0 \
  --dist_temperature 5.0 \
  --max_length 1024 \
  --per_device_train_batch_size 4 \
  --gradient_accumulation_steps 4 \
  --save_steps 100 \
  --eval_steps 100 \
  --logging_steps 10

echo "[INFO] Stage 2 complete. Weights saved to ./checkpoints/stage2_lora"
