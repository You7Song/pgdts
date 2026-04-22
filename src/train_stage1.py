#!/usr/bin/env python3
"""
PGDTS Stage 1 Training Entry Point
===================================
Scene Object Perception with Global Bag-of-Tokens Loss.

Usage:
    python src/train_stage1.py \
        --train_jsonl ./data/stage1_train.jsonl \
        --val_jsonl ./data/stage1_val.jsonl \
        --output_dir ./checkpoints/stage1_lora \
        --system_prompt "You are a professional scene perception module..."
"""

import sys
import os

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from src.trainers import Stage1Trainer
from src.trainers.pgdts_trainer_base import PGDTSArguments


def main():
    args = PGDTSArguments(
        model_id_or_path="Qwen/Qwen3-VL-8B-Instruct",
        train_jsonl="./data/stage1_train.jsonl",
        val_jsonl="./data/stage1_val.jsonl",
        output_dir="./checkpoints/stage1_lora",
        system_prompt="You are a professional scene perception module. Your core responsibility is to analyze the first-person navigation image and identify all objects that require attention for safe navigation. Focus on objects that could affect walking safety, path selection, or obstacle avoidance.",
        num_train_epochs=3,
        learning_rate=2e-4,
        lora_rank=24,
        lora_alpha=48,
        lambda_bag=0.5,
        alpha_bag=2.0,
        beta_bag=0.3,
        max_length=512,
    )
    trainer = Stage1Trainer(args)
    trainer.train()
    trainer.save_model(os.path.join(args.output_dir, "final"))
    print(f"[INFO] Stage 1 training complete. Model saved to {args.output_dir}")


if __name__ == "__main__":
    main()
