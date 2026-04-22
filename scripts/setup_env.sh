#!/bin/bash
# =============================================================================
# Environment Setup Script for PGDTS
# =============================================================================
# This script installs Python dependencies for the PGDTS project.
# PGDTS uses a unified training stack based on TRL + Transformers + PEFT.
#
# Usage:
#   chmod +x scripts/setup_env.sh
#   bash scripts/setup_env.sh
# =============================================================================

set -e

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
echo "Project root: ${PROJECT_ROOT}"

cd "${PROJECT_ROOT}"

# ---------------------------------------------------------------------------
# 1. Install Python dependencies
# ---------------------------------------------------------------------------
echo "[1/2] Installing Python dependencies from requirements.txt ..."
pip install -r requirements.txt

# ---------------------------------------------------------------------------
# 2. Download NLTK data (used by evaluation scripts)
# ---------------------------------------------------------------------------
echo "[2/2] Downloading NLTK resources ..."
python3 -c "import nltk; nltk.download('punkt', quiet=True); nltk.download('punkt_tab', quiet=True)"

echo ""
echo "========================================"
echo "Setup complete!"
echo "========================================"
echo ""
echo "Next steps:"
echo "  1. Download LoRA weights from Google Drive (see checkpoints/README.md)"
echo "  2. Verify image paths in train.jsonl / val.jsonl are valid on your system"
echo "  3. Run training:   bash scripts/train_stage3.sh"
echo "  4. Run inference:  bash scripts/infer.sh"
echo "  5. Run evaluation: bash scripts/eval.sh"
