#!/bin/bash
# =============================================================================
# Evaluation Pipeline
# =============================================================================
# Runs both surface-level text alignment metrics (TF-IDF, ROUGE) and
# functional evaluation (obstacle recognition, localization, navigation
# decision) using an external judge LLM.
#
# Usage:
#   chmod +x scripts/eval.sh
#   bash scripts/eval.sh [PREDICTIONS_JSONL] [GROUND_TRUTH_JSONL]
#
# Example:
#   bash scripts/eval.sh ./results/stage3_val_preds.jsonl ./data/val.jsonl
# =============================================================================

set -e

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "${PROJECT_ROOT}"

PREDICTIONS="${1:-./results/stage3_val_preds.jsonl}"
GROUND_TRUTH="${2:-./data/val.jsonl}"
RESULTS_DIR="./results"

mkdir -p ${RESULTS_DIR}

echo "========================================"
echo "PGDTS Evaluation Pipeline"
echo "========================================"
echo ""

# ---------------------------------------------------------------------------
# 1. Surface-Level Text Alignment Metrics
# ---------------------------------------------------------------------------
echo "[1/2] Computing TF-IDF and ROUGE scores ..."
python3 src/evaluate.py \
  --predictions ${PREDICTIONS} \
  --references ${GROUND_TRUTH} \
  --output ${RESULTS_DIR}/surface_metrics.json

echo "       Surface metrics saved to: ${RESULTS_DIR}/surface_metrics.json"
echo ""

# ---------------------------------------------------------------------------
# 2. Task Functional Evaluation (LLM-as-a-Judge)
# ---------------------------------------------------------------------------
echo "[2/2] Running functional evaluation with judge LLM ..."
echo "       This step requires an API key for the judge model (Qwen3.5-397B-A17B)."
echo "       Set your API key in the environment variable: JUDGE_API_KEY"
echo ""

python3 src/functional_eval.py \
  --predictions ${PREDICTIONS} \
  --references ${GROUND_TRUTH} \
  --output ${RESULTS_DIR}/functional_metrics.json \
  --api_key "${JUDGE_API_KEY:-}" \
  --api_base "${JUDGE_API_BASE:-https://dashscope.aliyuncs.com/compatible-mode/v1}" \
  --model "qwen3.5-397b-a17b" \
  --max_workers 4

echo "       Functional metrics saved to: ${RESULTS_DIR}/functional_metrics.json"
echo ""

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
echo "========================================"
echo "Evaluation Complete"
echo "========================================"
echo ""
echo "Results:"
echo "  - Surface metrics:    ${RESULTS_DIR}/surface_metrics.json"
echo "  - Functional metrics: ${RESULTS_DIR}/functional_metrics.json"
