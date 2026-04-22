#!/usr/bin/env python3
"""
PGDTS — Data Utilities
======================
Helper functions for inspecting, validating, and transforming the
multimodal JSONL datasets used in PGDTS.

These utilities are useful for debugging, analysis, and custom
preprocessing of the multimodal JSONL datasets.
"""

import json
import os
from collections import Counter
from typing import List, Dict, Any, Optional

from tqdm import tqdm


def load_jsonl(path: str) -> List[Dict[str, Any]]:
    """Load a JSONL file into a list of dictionaries."""
    data = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            data.append(json.loads(line))
    return data


def save_jsonl(path: str, data: List[Dict[str, Any]]) -> None:
    """Save a list of dictionaries to a JSONL file."""
    os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


def inspect_dataset(path: str, num_samples: int = 3) -> Dict[str, Any]:
    """Print a quick summary of the dataset."""
    data = load_jsonl(path)
    total = len(data)

    # Check image path existence
    missing_images = 0
    for item in tqdm(data, desc="Checking images"):
        for img in item.get("images", []):
            if not os.path.exists(img):
                missing_images += 1

    # Sample lengths
    user_lengths = []
    assistant_lengths = []
    for item in data:
        msgs = item.get("messages", [])
        for msg in msgs:
            if msg.get("role") == "user":
                user_lengths.append(len(msg.get("content", "")))
            elif msg.get("role") == "assistant":
                assistant_lengths.append(len(msg.get("content", "")))

    print(f"\nDataset: {path}")
    print(f"  Total samples      : {total}")
    print(f"  Missing images     : {missing_images}")
    print(f"  Avg user len       : {sum(user_lengths)/max(len(user_lengths),1):.1f}")
    print(f"  Avg assistant len  : {sum(assistant_lengths)/max(len(assistant_lengths),1):.1f}")
    print(f"\n--- Sample {num_samples} examples ---")
    for i, item in enumerate(data[:num_samples], 1):
        msgs = item.get("messages", [])
        images = item.get("images", [])
        print(f"\n[Sample {i}]")
        print(f"  Images: {images}")
        for msg in msgs:
            role = msg.get("role", "")
            content = msg.get("content", "")
            preview = content[:200].replace("\n", " ")
            print(f"  {role}: {preview}{'...' if len(content) > 200 else ''}")

    return {
        "total": total,
        "missing_images": missing_images,
        "avg_user_length": sum(user_lengths) / max(len(user_lengths), 1),
        "avg_assistant_length": sum(assistant_lengths) / max(len(assistant_lengths), 1),
    }


def add_system_prompt_to_dataset(
    input_path: str,
    output_path: str,
    system_prompt: str,
) -> None:
    """
    Inject a system prompt into every conversation in the dataset.
    Useful if you want to bake the system prompt into the JSONL instead
    of relying on the --system CLI argument.
    """
    data = load_jsonl(input_path)
    for item in data:
        msgs = item.get("messages", [])
        if msgs and msgs[0].get("role") == "system":
            msgs[0]["content"] = system_prompt
        else:
            msgs.insert(0, {"role": "system", "content": system_prompt})
        item["messages"] = msgs
    save_jsonl(output_path, data)
    print(f"[INFO] Saved {len(data)} samples with system prompt to: {output_path}")


def extract_decision_statistics(path: str) -> Dict[str, Any]:
    """Analyze decision texts for length distribution and common phrases."""
    data = load_jsonl(path)
    decisions = []
    for item in data:
        msgs = item.get("messages", [])
        if msgs and msgs[-1].get("role") == "assistant":
            decisions.append(msgs[-1]["content"].strip().lower())

    lengths = [len(d.split()) for d in decisions]
    counter = Counter()
    for d in decisions:
        counter.update(d.split())

    return {
        "num_decisions": len(decisions),
        "avg_words": sum(lengths) / max(len(lengths), 1),
        "min_words": min(lengths) if lengths else 0,
        "max_words": max(lengths) if lengths else 0,
        "top_tokens": counter.most_common(20),
    }


if __name__ == "__main__":
    import sys as _sys

    if len(_sys.argv) < 2:
        print("Usage: python data_utils.py <train.jsonl|val.jsonl>")
        _sys.exit(0)

    dataset_path = _sys.argv[1]
    inspect_dataset(dataset_path)
