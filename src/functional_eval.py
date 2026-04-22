#!/usr/bin/env python3
"""
PGDTS — Task Functional Evaluation (LLM-as-a-Judge)
=====================================================
Uses a high-capacity judge model (e.g., Qwen3.5-397B-A17B via DashScope/Bailian)
to score each prediction across three dimensions:

  1. Obstacle Recognition (0-5) : Correct identification of obstacles/target objects
  2. Obstacle Localization (0-5): Accuracy of spatial info (direction, distance, steps)
  3. Navigation Decision (0-5)  : Safety, rationality, and correctness of actions

The judge model receives a structured prompt with the image path, reference
answer, and model prediction, then outputs a JSON object with scores.

Reference: WalkVLM benchmark protocol adapted for PGDTS.
"""

import argparse
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict, Any, Optional

from tqdm import tqdm


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="PGDTS Functional Evaluation")
    parser.add_argument("--predictions", type=str, required=True,
                        help="Path to predictions JSONL")
    parser.add_argument("--references", type=str, required=True,
                        help="Path to ground-truth JSONL")
    parser.add_argument("--output", type=str, default="./results/functional_metrics.json")
    parser.add_argument("--api_key", type=str, required=True,
                        help="API key for the judge LLM service")
    parser.add_argument("--api_base", type=str,
                        default="https://dashscope.aliyuncs.com/compatible-mode/v1",
                        help="Base URL for the API endpoint")
    parser.add_argument("--model", type=str, default="qwen3.5-397b-a17b",
                        help="Judge model name (e.g., qwen3.5-397b-a17b)")
    parser.add_argument("--max_workers", type=int, default=4,
                        help="Number of concurrent API requests")
    parser.add_argument("--retry_times", type=int, default=3,
                        help="Max retries per request on failure")
    parser.add_argument("--temperature", type=float, default=0.0,
                        help="Temperature for judge model (0 for deterministic)")
    return parser.parse_args()


def load_jsonl(path: str) -> List[Dict[str, Any]]:
    data = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            data.append(json.loads(line))
    return data


def build_judge_prompt(image_path: str, reference: str, prediction: str, scene_data: str = "") -> str:
    """Construct the evaluation prompt for the judge LLM."""
    prompt = f"""You are an expert evaluator for assistive navigation systems for visually impaired users.
Your task is to compare the **Reference Answer** (ground truth) with the **Model Prediction** and score the prediction across three dimensions.

## Evaluation Dimensions (0-5 scale each)

1. **Obstacle Recognition** (0-5):
   - Does the prediction correctly identify the obstacles/target objects mentioned in the reference?
   - Deduct points for missing obstacles, hallucinated obstacles, or wrong object categories.

2. **Obstacle Localization** (0-5):
   - Does the prediction accurately describe spatial information (clock-face direction, distance in steps, relative position)?
   - Deduct points for wrong directions, wrong distances, or missing spatial cues.

3. **Navigation Decision** (0-5):
   - Is the suggested action safe, rational, and correct for the given scene?
   - Deduct points for dangerous advice, incorrect actions, or overly verbose/filler content.

## Input Data

- **Scene Context**: First-person navigation image at `{image_path}`
- **Scene Perception Data**:
{scene_data}

- **Reference Answer**: {reference}
- **Model Prediction**: {prediction}

## Output Format

Respond ONLY with a valid JSON object in the following format (no markdown fences, no explanations):

{{"obstacle_recognition": <int>, "obstacle_localization": <int>, "navigation_decision": <int>, "reasoning": "<brief reasoning>"}}
"""
    return prompt


def call_judge_api(
    api_key: str,
    api_base: str,
    model: str,
    prompt: str,
    temperature: float,
    retry_times: int = 3,
) -> Optional[Dict[str, Any]]:
    """
    Call the judge LLM API. Supports OpenAI-compatible interfaces
    (e.g., DashScope / Bailian / OpenAI).
    """
    try:
        from openai import OpenAI
    except ImportError:
        print("[ERROR] openai package is required for functional evaluation.")
        sys.exit(1)

    client = OpenAI(api_key=api_key, base_url=api_base)

    for attempt in range(retry_times):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": "You are a rigorous evaluator."},
                    {"role": "user", "content": prompt},
                ],
                temperature=temperature,
                max_tokens=512,
            )
            content = response.choices[0].message.content.strip()
            # Clean up markdown fences if present
            content = content.removeprefix("```json").removeprefix("```")
            content = content.removesuffix("```").strip()
            result = json.loads(content)
            # Validate keys
            assert all(k in result for k in ["obstacle_recognition", "obstacle_localization", "navigation_decision"])
            return result
        except Exception as e:
            print(f"[WARN] API call failed (attempt {attempt + 1}/{retry_times}): {e}")
            time.sleep(2 ** attempt)

    return None


def extract_scene_data(messages: List[Dict[str, str]]) -> str:
    """Extract the user message (scene perception data) from messages."""
    for msg in messages:
        if msg.get("role") == "user":
            content = msg.get("content", "")
            # Remove the <image> tag for readability in the prompt
            content = content.replace("<image>", "").strip()
            return content
    return ""


def evaluate_sample(
    pred_item: Dict[str, Any],
    ref_item: Dict[str, Any],
    api_key: str,
    api_base: str,
    model: str,
    temperature: float,
    retry_times: int,
) -> Dict[str, Any]:
    """Evaluate a single prediction-reference pair."""
    # Extract prediction text
    prediction = pred_item.get("prediction", "").strip()

    # Extract reference text
    ref_messages = ref_item.get("messages", [])
    reference = ""
    if ref_messages and ref_messages[-1].get("role") == "assistant":
        reference = ref_messages[-1]["content"].strip()

    # Extract image and scene data
    images = ref_item.get("images", [])
    image_path = images[0] if images else ""
    scene_data = extract_scene_data(ref_messages)

    prompt = build_judge_prompt(image_path, reference, prediction, scene_data)
    scores = call_judge_api(api_key, api_base, model, prompt, temperature, retry_times)

    if scores is None:
        scores = {
            "obstacle_recognition": 0,
            "obstacle_localization": 0,
            "navigation_decision": 0,
            "reasoning": "API call failed after retries",
        }

    return {
        "image": image_path,
        "reference": reference,
        "prediction": prediction,
        "scores": scores,
    }


def main():
    args = parse_args()

    if not args.api_key:
        print("[ERROR] --api_key is required for functional evaluation.")
        sys.exit(1)

    pred_data = load_jsonl(args.predictions)
    ref_data = load_jsonl(args.references)

    if len(pred_data) != len(ref_data):
        print(f"[WARN] Mismatch: preds={len(pred_data)}, refs={len(ref_data)}. Truncating to minimum.")
        min_len = min(len(pred_data), len(ref_data))
        pred_data = pred_data[:min_len]
        ref_data = ref_data[:min_len]

    results = []
    print(f"[INFO] Starting functional evaluation with {args.max_workers} workers ...")

    with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
        futures = {
            executor.submit(
                evaluate_sample,
                pred_data[i],
                ref_data[i],
                args.api_key,
                args.api_base,
                args.model,
                args.temperature,
                args.retry_times,
            ): i
            for i in range(len(pred_data))
        }

        for future in tqdm(as_completed(futures), total=len(futures), desc="Functional Eval"):
            i = futures[future]
            try:
                res = future.result()
                results.append(res)
            except Exception as e:
                print(f"[ERROR] Sample {i} evaluation failed: {e}")
                results.append({
                    "image": ref_data[i].get("images", [""])[0],
                    "reference": "",
                    "prediction": pred_data[i].get("prediction", ""),
                    "scores": {
                        "obstacle_recognition": 0,
                        "obstacle_localization": 0,
                        "navigation_decision": 0,
                        "reasoning": f"Exception: {e}",
                    },
                })

    # Sort results back to original order
    # (We lost order because of as_completed; restore via index if needed.
    #  For simplicity we just keep the list since mean is order-independent.)

    # Aggregate scores
    or_scores = [r["scores"]["obstacle_recognition"] for r in results]
    ol_scores = [r["scores"]["obstacle_localization"] for r in results]
    nd_scores = [r["scores"]["navigation_decision"] for r in results]
    total_scores = [a + b + c for a, b, c in zip(or_scores, ol_scores, nd_scores)]

    summary = {
        "num_samples": len(results),
        "obstacle_recognition": {
            "mean": sum(or_scores) / len(or_scores),
            "max": 5.0,
            "scores": or_scores,
        },
        "obstacle_localization": {
            "mean": sum(ol_scores) / len(ol_scores),
            "max": 5.0,
            "scores": ol_scores,
        },
        "navigation_decision": {
            "mean": sum(nd_scores) / len(nd_scores),
            "max": 5.0,
            "scores": nd_scores,
        },
        "total_score": {
            "mean": sum(total_scores) / len(total_scores),
            "max": 15.0,
            "scores": total_scores,
        },
        "per_sample": results,
    }

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print("\n========================================")
    print("Functional Evaluation Results")
    print("========================================")
    print(f"  Samples              : {summary['num_samples']}")
    print(f"  Obstacle Recognition : {summary['obstacle_recognition']['mean']:.2f} / 5.00")
    print(f"  Obstacle Localization: {summary['obstacle_localization']['mean']:.2f} / 5.00")
    print(f"  Navigation Decision  : {summary['navigation_decision']['mean']:.2f} / 5.00")
    print(f"  Total Score          : {summary['total_score']['mean']:.2f} / 15.00")
    print(f"\nSaved to: {args.output}")


if __name__ == "__main__":
    main()
