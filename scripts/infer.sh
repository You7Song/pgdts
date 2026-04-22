#!/bin/bash
# =============================================================================
# Batch Inference Script
# =============================================================================
# Runs inference on a JSONL dataset using the trained Stage 3 model and
# saves generated decisions.
#
# Usage:
#   chmod +x scripts/infer.sh
#   bash scripts/infer.sh [INPUT_JSONL] [OUTPUT_JSONL] [ADAPTER_DIR]
#
# Example:
#   bash scripts/infer.sh ./data/val.jsonl ./results/stage3_val_preds.jsonl ./checkpoints/stage3_lora
# =============================================================================

set -e

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "${PROJECT_ROOT}"

INPUT_JSONL="${1:-./data/val.jsonl}"
OUTPUT_JSONL="${2:-./results/stage3_val_preds.jsonl}"
ADAPTER_DIR="${3:-./checkpoints/stage3_lora}"
BASE_MODEL="Qwen/Qwen3-VL-8B-Instruct"

# Optional: set a system prompt (raw string or .txt file path).
# Leave empty if your JSONL already includes a system turn.
SYSTEM_PROMPT=""

mkdir -p "$(dirname ${OUTPUT_JSONL})"

echo "[INFO] Running inference"
echo "       Input    : ${INPUT_JSONL}"
echo "       Output   : ${OUTPUT_JSONL}"
echo "       Adapter  : ${ADAPTER_DIR}"

python3 src/inference.py \
  --model_id_or_path ${BASE_MODEL} \
  --adapter_path ${ADAPTER_DIR} \
  --input_jsonl ${INPUT_JSONL} \
  --output_jsonl ${OUTPUT_JSONL} \
  ${SYSTEM_PROMPT:+--system_prompt "${SYSTEM_PROMPT}"} \
  --max_new_tokens 256 \
  --temperature 0.1 \
  --top_p 0.9 \
  --batch_size 8

echo ""
echo "[INFO] Inference complete. Results saved to: ${OUTPUT_JSONL}"
