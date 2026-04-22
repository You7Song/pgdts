#!/usr/bin/env python3
"""
PGDTS Stage 3 — Batch Inference Script
======================================
Runs the trained Qwen3-VL model on a JSONL dataset and writes the
model-generated decisions to an output JSONL file.

Uses native transformers + PEFT for model loading and generation.
"""

import argparse
import json
import os
import sys
from typing import List, Dict, Any, Optional

from tqdm import tqdm

# Add project root to path for potential local imports
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="PGDTS Stage 3 Batch Inference")
    parser.add_argument("--model_id_or_path", type=str, required=True,
                        help="Base model ID or local path (e.g., Qwen/Qwen3-VL-8B-Instruct)")
    parser.add_argument("--adapter_path", type=str, default=None,
                        help="Path to LoRA adapter directory (optional)")
    parser.add_argument("--input_jsonl", type=str, required=True,
                        help="Input JSONL file with multimodal conversations")
    parser.add_argument("--output_jsonl", type=str, required=True,
                        help="Output JSONL file to save predictions")
    parser.add_argument("--system_prompt", type=str, default=None,
                        help="Path to system prompt txt file, or raw string")
    parser.add_argument("--max_new_tokens", type=int, default=256,
                        help="Maximum number of new tokens to generate")
    parser.add_argument("--temperature", type=float, default=0.1,
                        help="Sampling temperature")
    parser.add_argument("--top_p", type=float, default=0.9,
                        help="Nucleus sampling parameter")
    parser.add_argument("--top_k", type=int, default=20,
                        help="Top-k sampling parameter")
    parser.add_argument("--batch_size", type=int, default=1,
                        help="Inference batch size (use 1 for VLM to avoid collation issues)")
    parser.add_argument("--max_samples", type=int, default=None,
                        help="If set, only infer on the first N samples")
    return parser.parse_args()


def load_system_prompt(system_prompt_arg: Optional[str]) -> Optional[str]:
    """Load system prompt from file or return the string directly."""
    if not system_prompt_arg:
        return None
    if os.path.isfile(system_prompt_arg):
        with open(system_prompt_arg, "r", encoding="utf-8") as f:
            return f.read().strip()
    # If the argument looks like a file path but does not exist,
    # return None instead of accidentally using the path string as prompt text.
    if any(c in system_prompt_arg for c in ["/", "\\"]) or system_prompt_arg.endswith((".txt", ".md")):
        return None
    return system_prompt_arg.strip()


def load_jsonl(path: str) -> List[Dict[str, Any]]:
    data = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            data.append(json.loads(line))
    return data


def save_jsonl(path: str, data: List[Dict[str, Any]]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


def run_inference(
    model_id: str,
    adapter_path: Optional[str],
    samples: List[Dict[str, Any]],
    system_prompt: Optional[str],
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    top_k: int,
    batch_size: int,
) -> List[Dict[str, Any]]:
    """Run inference using transformers + PEFT."""
    import torch
    from transformers import AutoModelForVision2Seq, AutoProcessor
    from qwen_vl_utils import process_vision_info

    if adapter_path:
        from peft import PeftModel

    device = "cuda" if torch.cuda.is_available() else "cpu"
    processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
    model = AutoModelForVision2Seq.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        device_map="auto",
    )

    if adapter_path:
        model = PeftModel.from_pretrained(model, adapter_path)
        model = model.merge_and_unload()  # merge for speed

    model.eval()

    results = []
    for i in tqdm(range(0, len(samples), batch_size), desc="Inference"):
        batch = samples[i : i + batch_size]
        for sample in batch:
            messages = sample.get("messages", [])
            images = sample.get("images", [])

            if system_prompt and (not messages or messages[0].get("role") != "system"):
                messages = [{"role": "system", "content": system_prompt}] + messages

            # Format messages with embedded images for Qwen3-VL processor
            formatted_messages = []
            img_idx = 0
            for msg in messages:
                content = msg.get("content", "")
                if "<image>" in content:
                    parts = content.split("<image>")
                    new_content = []
                    for p_idx, part in enumerate(parts):
                        if p_idx > 0 and img_idx < len(images):
                            img_path = images[img_idx]
                            if os.path.exists(img_path):
                                from PIL import Image
                                new_content.append({"type": "image", "image": Image.open(img_path).convert("RGB")})
                            img_idx += 1
                        if part.strip() or p_idx == 0:
                            new_content.append({"type": "text", "text": part})
                    if new_content and new_content[-1].get("type") == "text" and not new_content[-1].get("text", "").strip():
                        new_content.pop()
                    formatted_messages.append({"role": msg["role"], "content": new_content})
                else:
                    formatted_messages.append(msg)

            text = processor.apply_chat_template(formatted_messages, tokenize=False, add_generation_prompt=True)
            image_inputs, video_inputs = process_vision_info(formatted_messages)
            inputs = processor(
                text=[text],
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt",
            )
            inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}

            with torch.no_grad():
                generated_ids = model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    top_k=top_k,
                    do_sample=temperature > 0,
                )
            generated_ids_trimmed = [
                out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs["input_ids"], generated_ids)
            ]
            pred_text = processor.batch_decode(
                generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )[0]

            sample_messages = sample.get("messages", [])
            gt_text = sample_messages[-1]["content"] if sample_messages and sample_messages[-1].get("role") == "assistant" else ""
            out_item = {
                "image": images[0] if images else None,
                "ground_truth": gt_text,
                "prediction": pred_text,
                "messages": sample_messages,
            }
            results.append(out_item)
    return results


def main():
    args = parse_args()
    system_prompt = load_system_prompt(args.system_prompt)
    print(f"[INFO] Loading data from: {args.input_jsonl}")
    samples = load_jsonl(args.input_jsonl)
    if args.max_samples:
        samples = samples[: args.max_samples]
    print(f"[INFO] Total samples to infer: {len(samples)}")

    results = run_inference(
        model_id=args.model_id_or_path,
        adapter_path=args.adapter_path,
        samples=samples,
        system_prompt=system_prompt,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        batch_size=args.batch_size,
    )

    save_jsonl(args.output_jsonl, results)
    print(f"[INFO] Saved {len(results)} predictions to: {args.output_jsonl}")


if __name__ == "__main__":
    main()
