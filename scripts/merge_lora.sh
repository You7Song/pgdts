#!/bin/bash
# =============================================================================
# Merge LoRA Adapter into Base Model
# =============================================================================
# Produces a standalone checkpoint that can be loaded without PEFT.
#
# Usage:
#   bash scripts/merge_lora.sh [ADAPTER_DIR] [OUTPUT_DIR]
# =============================================================================

set -e

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "${PROJECT_ROOT}"

ADAPTER_DIR="${1:-./checkpoints/stage3_lora}"
OUTPUT_DIR="${2:-./checkpoints/stage3_merged}"
BASE_MODEL="Qwen/Qwen3-VL-8B-Instruct"

echo "[INFO] Merging LoRA adapter"
echo "       Base model : ${BASE_MODEL}"
echo "       Adapter    : ${ADAPTER_DIR}"
echo "       Output     : ${OUTPUT_DIR}"

python3 src/merge_lora.py \
  --model_id_or_path ${BASE_MODEL} \
  --adapter_path ${ADAPTER_DIR} \
  --output_dir ${OUTPUT_DIR} \
  --torch_dtype bfloat16

echo "[INFO] Merge complete. Merged model saved to: ${OUTPUT_DIR}"
