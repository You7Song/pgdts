#!/usr/bin/env python3
"""
PGDTS Stage 2 Training Entry Point
===================================
Spatial Understanding with Distance Regression Loss.

Usage:
    python src/train_stage2.py \
        --train_jsonl ./data/stage2_train.jsonl \
        --val_jsonl ./data/stage2_val.jsonl \
        --adapter_path ./checkpoints/stage1_lora \
        --output_dir ./checkpoints/stage2_lora
"""

import sys
import os

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from src.trainers import Stage2Trainer
from src.trainers.pgdts_trainer_base import PGDTSArguments


def main():
    args = PGDTSArguments(
        model_id_or_path="Qwen/Qwen3-VL-8B-Instruct",
        adapter_path="./checkpoints/stage1_lora",
        train_jsonl="./data/stage2_train.jsonl",
        val_jsonl="./data/stage2_val.jsonl",
        output_dir="./checkpoints/stage2_lora",
        system_prompt="You are a professional spatial understanding module. Your core responsibility is to analyze the navigation scene and determine the clock-face direction and absolute distance for each relevant object. Use clock-face directions (12 o'clock is directly ahead, 3 o'clock is to the right, etc.) and provide distance in steps (1 step ≈ 0.6 meters).",
        num_train_epochs=4,
        learning_rate=1.5e-4,
        lora_rank=24,
        lora_alpha=48,
        lambda_dist=0.2,
        huber_delta=1.0,
        dist_temperature=5.0,
        max_length=1024,
    )
    trainer = Stage2Trainer(args)
    trainer.train()
    trainer.save_model(os.path.join(args.output_dir, "final"))
    print(f"[INFO] Stage 2 training complete. Model saved to {args.output_dir}")


if __name__ == "__main__":
    main()
