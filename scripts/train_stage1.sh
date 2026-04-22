#!/bin/bash
# =============================================================================
# Stage 1 Training Script: Scene Object Perception
# =============================================================================
# Trains the model to identify navigation-relevant objects from first-person
# images using the Global Bag-of-Tokens Loss (L_Bag) — order-invariant
# multi-label classification over the vocabulary.
#
# Loss: L_Stage1 = L_AR + λ₁ · L_Bag   (λ₁=0.5, α=2.0, β=0.3)
#
# Data format (JSONL):
#   {
#     "messages": [...],
#     "images": [...],
#     "object_list": ["vehicle", "pedestrian"]   # optional; auto-parsed if absent
#   }
#
# Usage:
#   bash scripts/train_stage1.sh
# =============================================================================

set -e

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "${PROJECT_ROOT}"

echo "[INFO] Stage 1 — Scene Object Perception (Bag-of-Tokens Loss)"

python3 src/train_stage1.py \
  --train_jsonl ./data/stage1_train.jsonl \
  --val_jsonl ./data/stage1_val.jsonl \
  --output_dir ./checkpoints/stage1_lora \
  --num_train_epochs 3 \
  --learning_rate 2e-4 \
  --lora_rank 24 \
  --lora_alpha 48 \
  --lambda_bag 0.5 \
  --alpha_bag 2.0 \
  --beta_bag 0.3 \
  --max_length 512 \
  --per_device_train_batch_size 4 \
  --gradient_accumulation_steps 4 \
  --save_steps 100 \
  --eval_steps 100 \
  --logging_steps 10

echo "[INFO] Stage 1 complete. Weights saved to ./checkpoints/stage1_lora"
