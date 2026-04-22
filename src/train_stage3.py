#!/usr/bin/env python3
"""
PGDTS Stage 3 Training Entry Point
===================================
Decision Generation via standard autoregressive SFT.

Usage:
    python src/train_stage3.py \
        --train_jsonl ./data/train.jsonl \
        --val_jsonl ./data/val.jsonl \
        --system_prompt "Your system prompt here..." \
        --adapter_path ./checkpoints/stage2_lora \
        --output_dir ./checkpoints/stage3_lora
"""

import sys
import os

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from src.trainers import Stage3Trainer
from src.trainers.pgdts_trainer_base import PGDTSArguments


def main():
    args = PGDTSArguments(
        model_id_or_path="Qwen/Qwen3-VL-8B-Instruct",
        adapter_path="./checkpoints/stage2_lora",
        train_jsonl="./data/train.jsonl",
        val_jsonl="./data/val.jsonl",
        output_dir="./checkpoints/stage3_lora",
        system_prompt=None,  # set to a string or a .txt file path
        num_train_epochs=2,
        learning_rate=1e-4,
        lora_rank=24,
        lora_alpha=48,
        max_length=2048,
    )
    trainer = Stage3Trainer(args)
    trainer.train()
    trainer.save_model(os.path.join(args.output_dir, "final"))
    print(f"[INFO] Stage 3 training complete. Model saved to {args.output_dir}")


if __name__ == "__main__":
    main()
