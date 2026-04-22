#!/usr/bin/env python3
"""
Merge LoRA Adapter into Base Model
====================================
Produces a standalone (merged) checkpoint that can be loaded without PEFT,
enabling faster inference and easier deployment.

Usage:
    python src/merge_lora.py \
        --model_id_or_path Qwen/Qwen3-VL-8B-Instruct \
        --adapter_path ./checkpoints/stage3_lora \
        --output_dir ./checkpoints/stage3_merged
"""

import argparse
import os

import torch
from peft import PeftModel
from transformers import AutoModelForVision2Seq, AutoProcessor


def parse_args():
    parser = argparse.ArgumentParser(description="Merge LoRA adapter into base model")
    parser.add_argument("--model_id_or_path", type=str, required=True, help="Base model HF id or local path")
    parser.add_argument("--adapter_path", type=str, required=True, help="Path to LoRA adapter directory")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory for merged model")
    parser.add_argument("--torch_dtype", type=str, default="bfloat16", choices=["bfloat16", "float16", "float32"])
    return parser.parse_args()


def main():
    args = parse_args()
    dtype_map = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}
    torch_dtype = dtype_map[args.torch_dtype]

    print(f"[INFO] Loading base model: {args.model_id_or_path}")
    model = AutoModelForVision2Seq.from_pretrained(
        args.model_id_or_path,
        torch_dtype=torch_dtype,
        trust_remote_code=True,
        device_map="cpu",  # load on CPU to avoid OOM during merge
    )
    processor = AutoProcessor.from_pretrained(args.model_id_or_path, trust_remote_code=True)

    print(f"[INFO] Loading adapter: {args.adapter_path}")
    model = PeftModel.from_pretrained(model, args.adapter_path)

    print("[INFO] Merging LoRA weights into base model ...")
    model = model.merge_and_unload()

    print(f"[INFO] Saving merged model to: {args.output_dir}")
    os.makedirs(args.output_dir, exist_ok=True)
    model.save_pretrained(args.output_dir)
    processor.save_pretrained(args.output_dir)

    print("[INFO] Merge complete.")


if __name__ == "__main__":
    main()
