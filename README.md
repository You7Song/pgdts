<h1 align="center">PGDTS</h1>
<p align="center"><b>P</b>rogressive <b>P</b>erception-<b>G</b>uided <b>D</b>ecision <b>T</b>raining <b>S</b>trategy</p>
<p align="center">面向导盲避险的渐进式感知引导决策训练方法</p>

<p align="center">
  <a href="#-quick-start">Quick Start</a> •
  <a href="#-methodology">Methodology</a> •
  <a href="#-training">Training</a> •
  <a href="#-inference">Inference</a> •
  <a href="#-evaluation">Evaluation</a>
</p>

---

## 📌 Overview

PGDTS addresses a critical challenge in **intelligent assistive navigation for the visually impaired**: mainstream Multimodal Large Language Models (MLLMs) struggle with fine-grained visual perception and structured spatial reasoning, leading to unsafe navigation decisions.

Our solution adopts a **Decoupled Perception-Decision Architecture**:
- **Front-end**: Specialized vision models (YOLOE, SAM 3, Depth-Pro) extract high-precision structured perception.
- **Back-end**: A lightweight MLLM (**Qwen3-VL-8B-Instruct**) focuses purely on **decision-making**, leveraging structured inputs through hierarchical perception-decision association.

Through **three progressive SFT stages**, PGDTS builds this association step-by-step, achieving substantial improvements over direct end-to-end fine-tuning:

| Metric | Qwen3-VL-8B + SFT | **PGDTS (Ours)** | Δ |
|--------|-------------------|------------------|---|
| **TF-IDF** | 0.261 | **0.551** | **+0.290** |
| **ROUGE-1** | 0.437 | **0.580** | +0.143 |
| **ROUGE-L** | 0.390 | **0.521** | **+0.131** |
| **Functional Score** | 9.28 | **11.16** | **+20%** |

---

## 🏗️ Methodology

### Decoupled Perception-Decision Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                         INFERENCE PIPELINE                       │
├─────────────────────────────────────────────────────────────────┤
│  Input Image                                                     │
│      │                                                           │
│      ▼                                                           │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────────────┐  │
│  │  YOLOE-26x  │───▶│  SAM 3      │───▶│  Depth-Pro          │  │
│  │  (Object    │    │  (Mask)     │    │  (Absolute Depth)   │  │
│  │  Detection) │    │             │    │                     │  │
│  └─────────────┘    └─────────────┘    └─────────────────────┘  │
│         │                  │                      │              │
│         └──────────────────┼──────────────────────┘              │
│                            ▼                                     │
│              Structured Perception Data                          │
│              • Object list + category                            │
│              • Clock-face direction                              │
│              • Distance in steps (~0.6m/step)                    │
│                            │                                     │
│                            ▼                                     │
│  ┌───────────────────────────────────────────────────────────┐  │
│  │              Qwen3-VL-8B-Instruct + LoRA                   │  │
│  │         (Decision Core: generates avoidance instruction)   │  │
│  └───────────────────────────────────────────────────────────┘  │
│                            │                                     │
│                            ▼                                     │
│              "Vehicle approaching from 12 o'clock, 2 steps,      │
│               stop and wait."                                    │
└─────────────────────────────────────────────────────────────────┘
```

### Three-Stage Progressive Training

| Stage | Name | Input | Output | Loss | Epochs |
|-------|------|-------|--------|------|--------|
| **1** | Scene Object Perception | Image only | Relevant object list | L_AR + λ₁·L_Bag (bag-of-tokens) | 3 |
| **2** | Spatial Understanding | Image + Object list | Direction & distance per object | L_AR + λ_dist·L_dist (Huber regression) | 4 |
| **3** | **Decision Generation** | Image + Structured perception | Concise avoidance instruction | **L_AR (standard SFT)** | **2** |

> **This repository focuses on Stage 3**, as it is the final decision-generation phase. The expected training data (`train.jsonl` / `val.jsonl`) contains structured perception outputs (simulating Stage 1 & 2 results), allowing the model to learn the mapping from "perception → decision". Refer to `data/README.md` for the data format specification.

---

## 🚀 Quick Start

### 1. Environment Setup

```bash
# Clone this repo
git clone <your-github-repo-url> pgdts
cd pgdts

# Install dependencies (TRL + Transformers + PEFT)
bash scripts/setup_env.sh
```

> The base model (`Qwen/Qwen3-VL-8B-Instruct`) will be downloaded automatically on first use by `transformers`.

### 2. Download Pre-trained LoRA Weights

Download the LoRA weights from Google Drive and place them under `checkpoints/`:

🔗 **Link**: https://drive.google.com/drive/folders/19QDIqjbl85Ua7KTOa4q4NKTn4ohgcbqt?usp=drive_link

Expected structure:
```
checkpoints/
├── stage2_lora/       # (optional) Stage 2 adapter for warm-starting Stage 3
└── stage3_lora/       # Final Stage 3 adapter
```

See [`checkpoints/README.md`](checkpoints/README.md) for detailed download instructions.

---

## 🏋️ Training

All training is built on **TRL** (`SFTTrainer`) + **Transformers** + **PEFT**, with custom `compute_loss` overrides for Stage 1 & 2.

### Stage 1: Scene Object Perception

**Objective**: Learn visual-semantic association for navigation-relevant objects.

**Loss**: `L_Stage1 = L_AR + λ₁·L_Bag`

- **L_AR**: Standard autoregressive cross-entropy.
- **L_Bag**: Global Bag-of-Tokens Loss — time-dimension max-pooled, order-invariant multi-label classification over the vocabulary (α=2.0, β=0.3, λ₁=0.5).

```bash
python3 src/train_stage1.py \
  --train_jsonl ./data/stage1_train.jsonl \
  --val_jsonl ./data/stage1_val.jsonl \
  --output_dir ./checkpoints/stage1_lora
```

> **Data format**: Each JSONL line should contain `messages`, `images`, and optionally `object_list: ["vehicle", "pedestrian", ...]`. If `object_list` is absent, the trainer auto-parses it from the assistant text via regex (`Relevant Objects: ...`).

### Stage 2: Spatial Understanding

**Objective**: Predict clock-face direction and absolute distance (steps) per object.

**Loss**: `L_Stage2 = L_AR + λ_dist·L_dist`

- **L_AR**: Standard autoregressive cross-entropy.
- **L_dist**: Token-level Distance Regression Loss — a lightweight linear head (`d+1` params) on the hidden state at each distance-numeric token, with distance-aware Huber weighting (δ=1.0, τ=5, λ_dist=0.2).

```bash
python3 src/train_stage2.py \
  --train_jsonl ./data/stage2_train.jsonl \
  --val_jsonl ./data/stage2_val.jsonl \
  --adapter_path ./checkpoints/stage1_lora \
  --output_dir ./checkpoints/stage2_lora
```

> **Data format**: Each JSONL line should contain `messages`, `images`, and optionally `spatial_info` with `distance_steps` and `distance_token_text`. If absent, distances are auto-parsed from assistant text (e.g., `"2 steps"`).

### Stage 3: Decision Generation

**Objective**: Generate concise, actionable avoidance instructions.

**Loss**: `L_Stage3 = L_AR^decision` (standard SFT)

```bash
bash scripts/train_stage3.sh
```

Or directly:
```bash
python3 src/train_stage3.py \
  --train_jsonl ./data/train.jsonl \
  --val_jsonl ./data/val.jsonl \
  --adapter_path ./checkpoints/stage2_lora \
  --output_dir ./checkpoints/stage3_lora
```

> **Note**: Adjust `--train_jsonl` / `--val_jsonl` paths to your local dataset. If your JSONL already includes a system turn, omit `--system_prompt`; otherwise pass a raw string or a `.txt` file path.

| Parameter | Value | Description |
|-----------|-------|-------------|
| `model` | `Qwen/Qwen3-VL-8B-Instruct` | Base MLLM |
| `lora_rank` | `24` | LoRA rank |
| `lora_alpha` | `48` | LoRA alpha |
| `num_train_epochs` | `2` | Stage 3 epochs |
| `learning_rate` | `1e-4` | Peak LR (cosine) |
| `weight_decay` | `0.01` | Weight decay |

If `checkpoints/stage2_lora/` exists, the script automatically loads it as the initialization for Stage 3 (iterative weight passing). Otherwise, it trains from the base model.

### Custom Loss Implementation Details

| Stage | File | Custom Component | Paper Equation |
|-------|------|------------------|----------------|
| 1 | `src/trainers/stage1_trainer.py` | `Stage1Trainer.compute_loss` — multi-hot vocab max-pool + weighted BCE | Eq. 4.1–4.4 |
| 2 | `src/trainers/stage2_trainer.py` | `Stage2Trainer.compute_loss` — token-level `nn.Linear` regression head + distance-weighted Huber | Eq. 4.7–4.9 |
| 3 | `src/trainers/stage3_trainer.py` | Standard AR loss (inherited from base) | Eq. 4.10–4.11 |

All three trainers share the same base class (`PGDTSTrainerBase`) which extends TRL's `SFTTrainer` and handles model loading, LoRA configuration, dataset preparation, and Qwen3-VL image processing.

---

## 🔍 Inference

### Batch Inference

```bash
bash scripts/infer.sh ./data/val.jsonl ./results/stage3_val_preds.jsonl ./checkpoints/stage3_lora
```

> Replace `./data/val.jsonl` with your own inference dataset.

This will:
1. Load the base model + Stage 3 LoRA adapter
2. Run inference on all samples in the provided JSONL
3. Save predictions to `results/stage3_val_preds.jsonl`

### Interactive Single-Image Inference

```python
from src.inference import run_inference

results = run_inference(
    model_id="Qwen/Qwen3-VL-8B-Instruct",
    adapter_path="./checkpoints/stage3_lora",
    samples=[...],  # list of dicts with messages & images
    system_prompt="...",
    max_new_tokens=256,
    temperature=0.1,
)
```

---

## 📊 Evaluation

### Full Evaluation Pipeline

```bash
bash scripts/eval.sh ./results/stage3_val_preds.jsonl ./data/val.jsonl
```

> Replace `./data/val.jsonl` with your own ground-truth dataset.

This runs two evaluation protocols:

#### 1. Surface-Level Text Alignment

Computed automatically by `src/evaluate.py`:

- **TF-IDF**: Cosine similarity of TF-IDF vectors
- **ROUGE-1 / ROUGE-2 / ROUGE-L**: N-gram and LCS overlap

Results are saved to `results/surface_metrics.json`.

#### 2. Task Functional Evaluation (LLM-as-a-Judge)

Computed by `src/functional_eval.py` using a high-capacity judge model (e.g., **Qwen3.5-397B-A17B** via Alibaba Cloud Bailian / DashScope):

| Dimension | Range | What it measures |
|-----------|-------|------------------|
| Obstacle Recognition | 0–5 | Correct identification of risk objects |
| Obstacle Localization | 0–5 | Accuracy of direction & distance |
| Navigation Decision | 0–5 | Safety & actionability of advice |

**Requirements**: Set your API key as an environment variable:

```bash
export JUDGE_API_KEY="sk-xxxxxxxx"
export JUDGE_API_BASE="https://dashscope.aliyuncs.com/compatible-mode/v1"
```

Results are saved to `results/functional_metrics.json`.

---

## 📁 Project Structure

```
pgdts/
├── .gitignore
├── README.md                          # This file
├── requirements.txt                   # Python dependencies (TRL + Transformers + PEFT)
├── scripts/
│   ├── setup_env.sh                   # Environment setup
│   ├── train_stage1.sh               # Stage 1 training (Bag-of-Tokens Loss)
│   ├── train_stage2.sh               # Stage 2 training (Distance Regression Loss)
│   ├── train_stage3.sh               # ⭐ Stage 3 training (Decision Generation SFT)
│   ├── merge_lora.sh                 # Merge LoRA into base model
│   ├── infer.sh                      # Batch inference wrapper
│   └── eval.sh                       # Evaluation pipeline wrapper
├── src/
│   ├── trainers/
│   │   ├── __init__.py               # Trainer exports
│   │   ├── pgdts_trainer_base.py     # Base class (extends TRL SFTTrainer)
│   │   ├── stage1_trainer.py         # Stage 1: Bag-of-Tokens Loss
│   │   ├── stage2_trainer.py         # Stage 2: Distance Regression + Huber
│   │   └── stage3_trainer.py         # Stage 3: Standard SFT
│   ├── train_stage1.py               # Stage 1 entry point
│   ├── train_stage2.py               # Stage 2 entry point
│   ├── train_stage3.py               # Stage 3 entry point
│   ├── inference.py                  # Batch inference implementation
│   ├── evaluate.py                   # TF-IDF + ROUGE evaluation
│   ├── functional_eval.py            # LLM-as-a-Judge functional evaluation
│   └── data_utils.py                 # Dataset inspection utilities
├── configs/
│   └── stage3_config.yaml            # Hyperparameter documentation
├── data/
│   └── README.md                     # Data format documentation
└── checkpoints/
    ├── stage1_lora/                  # (generated/download) Stage 1 adapter
    ├── stage2_lora/                  # (generated/download) Stage 2 adapter
    ├── stage3_lora/                  # (generated/download) Stage 3 final adapter
    ├── stage3_merged/                # (generated) Merged full model
    └── README.md                     # Weight download instructions
```

---

## 📝 Data Format

The training/validation data follows a **multimodal conversation JSONL format** compatible with Qwen3-VL:

```json
{
  "messages": [
    {
      "role": "user",
      "content": "<image>\nScene Perception Data:\n- Scene Category: Urban Street Intersection\n- Relevant Obstacles:\n  1. pedestrians (Direction: 12 o'clock, Distance: 5 steps)\n  2. vehicles (Direction: 11 o'clock, Distance: 5 steps)\n\nPlease provide concise and essential navigation prompts:"
    },
    {
      "role": "assistant",
      "content": "there are pedestrians passing five steps ahead, and there are passing vehicles at 11 o'clock. please keep moving forward on the current road."
    }
  ],
  "images": ["/path/to/first_person_frame.jpg"]
}
```

- `messages`: OpenAI-style conversation turns.
- `images`: Absolute paths to the corresponding first-person images.
- `<image>`: Placeholder token processed by the Qwen3-VL `AutoProcessor`.

> ⚠️ **Image Path Notice**: When preparing your own dataset, ensure the `images` field contains valid absolute paths on your system. If you obtain data from an external source (e.g., WalkVLM) and the image root differs from your local setup, bulk-replace the prefix using `sed` or a simple Python script:
>
> ```bash
> # Example: update paths to your local image directory
> sed -i 's|/old/path/to/images|/your/local/path/to/images|g' data/train.jsonl data/val.jsonl
> ```

---

## 🎯 Reproducing Paper Results

To reproduce the quantitative results from the paper:

1. **Train Stage 3** (or use the provided LoRA weights):
   ```bash
   bash scripts/train_stage3.sh
   ```

2. **Run inference** on your dataset:
   ```bash
   bash scripts/infer.sh ./data/val.jsonl ./results/preds.jsonl ./checkpoints/stage3_lora
   ```

3. **Evaluate** against ground truth:
   ```bash
   export JUDGE_API_KEY="your-api-key"
   bash scripts/eval.sh ./results/preds.jsonl ./data/val.jsonl
   ```

4. **Check results**:
   ```bash
   cat results/surface_metrics.json
   cat results/functional_metrics.json
   ```

Expected approximate scores on the WalkVLM Reminder Task:
- **TF-IDF**: ~0.55
- **ROUGE-L**: ~0.52
- **Functional Total**: ~11.2 / 15.0

---

## 📄 Citation

If you find this work useful, please consider citing:

```bibtex
@article{pgdts2025,
  title={Progressive Perception-Guided Decision Training for Intelligent Assistive Navigation for the Visually Impaired},
  author={Your Name},
  journal={},
  year={2025}
}
```

---

## 📜 License

This project is released under the MIT License. The base model (Qwen3-VL-8B-Instruct) follows its original license (Qwen License).

The underlying image data is derived from the **WalkVLM** dataset; please refer to WalkVLM's license for usage terms.

---

## 🙏 Acknowledgements

- [TRL](https://github.com/huggingface/trl) — Transformer Reinforcement Learning library
- [Transformers](https://github.com/huggingface/transformers) — Model inference and training
- [PEFT](https://github.com/huggingface/peft) — Parameter-Efficient Fine-Tuning
- [Qwen3-VL](https://github.com/QwenLM/Qwen3-VL) — Base vision-language model
- [WalkVLM](https://walkvlm2024.github.io/) — A diverse walking awareness dataset with 12k video-manual annotation pairs from Europe and Asia, providing a fair benchmark for blind walking tasks.
