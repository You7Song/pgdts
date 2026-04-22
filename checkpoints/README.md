# Model Checkpoints

This directory is used to store model weights and LoRA adapters.

## Pre-trained LoRA Weights

The LoRA weights for the PGDTS progressive training stages are hosted on Google Drive:

🔗 **Google Drive Link**: https://drive.google.com/drive/folders/19QDIqjbl85Ua7KTOa4q4NKTn4ohgcbqt?usp=drive_link

### Expected Directory Structure After Download

```
checkpoints/
├── stage1_lora/          # Stage 1: Scene Object Perception LoRA
│   ├── adapter_config.json
│   └── adapter_model.safetensors
├── stage2_lora/          # Stage 2: Spatial Understanding LoRA
│   ├── adapter_config.json
│   └── adapter_model.safetensors
└── stage3_lora/          # Stage 3: Decision Generation LoRA (final)
    ├── adapter_config.json
    └── adapter_model.safetensors
```

### How to Download

You can download the weights manually from the Google Drive link above, or use `gdown`:

```bash
pip install gdown
# Download the entire folder (replace FOLDER_ID with the actual folder ID from the link)
gdown --folder https://drive.google.com/drive/folders/19QDIqjbl85Ua7KTOa4q4NKTn4ohgcbqt?usp=drive_link -O ./checkpoints/
```

### Usage

**Stage 3 Training (continue from Stage 2)**:
```bash
python3 src/train_stage3.py \
  --train_jsonl ./data/train.jsonl \
  --val_jsonl ./data/val.jsonl \
  --adapter_path ./checkpoints/stage2_lora \
  --output_dir ./checkpoints/stage3_lora
```

> Refer to `data/README.md` for the expected data format when running training or inference.

**Inference with Final Weights**:
```bash
# Merge LoRA first (optional but recommended for faster inference)
python3 src/merge_lora.py \
  --model_id_or_path Qwen/Qwen3-VL-8B-Instruct \
  --adapter_path ./checkpoints/stage3_lora \
  --output_dir ./checkpoints/stage3_merged

# Or run inference directly with adapter
python3 src/inference.py \
  --model_id_or_path Qwen/Qwen3-VL-8B-Instruct \
  --adapter_path ./checkpoints/stage3_lora \
  --input_jsonl ./data/val.jsonl \
  --output_jsonl ./results/preds.jsonl
```

## Base Model

The base model is **Qwen3-VL-8B-Instruct** from Hugging Face:
- Model ID: `Qwen/Qwen3-VL-8B-Instruct`
- It will be automatically downloaded by `transformers` on first use.
