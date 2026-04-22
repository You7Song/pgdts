"""
PGDTS Trainer Base
==================
Base classes for the three-stage progressive training pipeline.
Handles model loading (Qwen3-VL + LoRA), dataset preparation,
and data collation for multimodal JSONL datasets.

This module uses TRL (SFTTrainer) + Transformers + PEFT to enable
full customisation of loss functions for Stage 1 (Bag-of-Tokens)
and Stage 2 (Distance Regression).
"""

import json
import os
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from peft import LoraConfig, get_peft_model, PeftModel
from PIL import Image
from torch.utils.data import Dataset
from transformers import (
    AutoModelForVision2Seq,
    AutoProcessor,
    TrainingArguments,
)
from trl import SFTTrainer
from transformers.trainer_utils import EvalPrediction

# qwen_vl_utils is required for Qwen3-VL image processing
try:
    from qwen_vl_utils import process_vision_info
except ImportError:
    process_vision_info = None


# ---------------------------------------------------------------------------
# Configuration dataclass
# ---------------------------------------------------------------------------

@dataclass
class PGDTSArguments:
    """Extra arguments shared across all three PGDTS stages."""

    # Paths
    train_jsonl: str = field(metadata={"help": "Path to training JSONL"})
    val_jsonl: Optional[str] = field(default=None, metadata={"help": "Path to validation JSONL"})
    output_dir: str = field(default="./checkpoints", metadata={"help": "Output directory"})
    system_prompt: Optional[str] = field(default=None, metadata={"help": "System prompt text or .txt file path"})

    # Model
    model_id_or_path: str = field(default="Qwen/Qwen3-VL-8B-Instruct", metadata={"help": "Base model HF id or local path"})
    adapter_path: Optional[str] = field(default=None, metadata={"help": "Path to existing LoRA adapter to resume from"})

    # LoRA
    lora_rank: int = field(default=24)
    lora_alpha: int = field(default=48)
    lora_dropout: float = field(default=0.0)
    lora_target_modules: Optional[List[str]] = field(
        default_factory=lambda: [
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
            "fc1", "fc2",           # visual merger MLP
        ]
    )

    # Training hyper-params (override TrainingArguments defaults)
    num_train_epochs: float = field(default=2.0)
    per_device_train_batch_size: int = field(default=4)
    per_device_eval_batch_size: int = field(default=4)
    gradient_accumulation_steps: int = field(default=4)
    learning_rate: float = field(default=1e-4)
    lr_scheduler_type: str = field(default="cosine")
    weight_decay: float = field(default=0.01)
    warmup_ratio: float = field(default=0.05)
    max_length: int = field(default=2048)
    bf16: bool = field(default=True)
    fp16: bool = field(default=False)
    save_steps: int = field(default=100)
    eval_steps: int = field(default=100)
    logging_steps: int = field(default=10)
    save_total_limit: int = field(default=3)
    seed: int = field(default=42)
    dataloader_num_workers: int = field(default=4)
    remove_unused_columns: bool = field(default=False)
    group_by_length: bool = field(default=False)

    # Stage-specific
    lambda_bag: float = field(default=0.5, metadata={"help": "Weight for Stage 1 bag-of-tokens loss"})
    alpha_bag: float = field(default=2.0, metadata={"help": "Stage 1 bag loss positive sample weight"})
    beta_bag: float = field(default=0.3, metadata={"help": "Stage 1 bag loss negative sample weight"})
    lambda_dist: float = field(default=0.2, metadata={"help": "Weight for Stage 2 distance loss"})
    huber_delta: float = field(default=1.0, metadata={"help": "Huber delta for Stage 2 (in steps)"})
    dist_temperature: float = field(default=5.0, metadata={"help": "Distance-aware weight temperature τ"})


def load_system_prompt(sp: Optional[str]) -> Optional[str]:
    if not sp:
        return None
    if os.path.isfile(sp):
        with open(sp, "r", encoding="utf-8") as f:
            return f.read().strip()
    # If the argument looks like a file path but does not exist,
    # return None instead of accidentally using the path string as prompt text.
    if any(c in sp for c in ["/", "\\"]) or sp.endswith((".txt", ".md")):
        return None
    return sp.strip()


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class PGDTSDataset(Dataset):
    """
    Loads multimodal JSONL and lazily prepares text+image inputs.
    Subclasses override `_extract_labels` to provide stage-specific tensors.
    """

    def __init__(
        self,
        jsonl_path: str,
        processor: AutoProcessor,
        system_prompt: Optional[str] = None,
        max_length: int = 2048,
    ):
        self.processor = processor
        self.tokenizer = processor.tokenizer
        self.system_prompt = system_prompt
        self.max_length = max_length
        self.samples: List[Dict[str, Any]] = []

        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    self.samples.append(json.loads(line))

    def __len__(self):
        return len(self.samples)

    def _build_messages(self, sample: Dict[str, Any]) -> List[Dict[str, str]]:
        """Build the message list, injecting system prompt if needed."""
        messages = sample.get("messages", [])
        # Defensive copy
        messages = [dict(m) for m in messages]
        if self.system_prompt:
            if messages and messages[0].get("role") == "system":
                messages[0]["content"] = self.system_prompt
            else:
                messages.insert(0, {"role": "system", "content": self.system_prompt})
        return messages

    def _load_images(self, images: List[str]) -> List[Image.Image]:
        """Load images from paths, warning if any are missing."""
        loaded = []
        for img_path in images:
            if os.path.exists(img_path):
                loaded.append(Image.open(img_path).convert("RGB"))
            else:
                import warnings
                warnings.warn(f"Image not found, using blank placeholder: {img_path}")
                loaded.append(Image.new("RGB", (224, 224), color=(0, 0, 0)))
        return loaded

    def _format_messages(self, messages: List[Dict[str, str]], loaded_images: List[Image.Image]):
        """Convert ms-swift style <image> placeholders into Qwen3-VL content blocks."""
        formatted_messages = []
        img_idx = 0
        for msg in messages:
            content = msg.get("content", "")
            if "<image>" in content:
                parts = content.split("<image>")
                new_content = []
                for i, part in enumerate(parts):
                    if i > 0 and img_idx < len(loaded_images):
                        new_content.append({"type": "image", "image": loaded_images[img_idx]})
                        img_idx += 1
                    if part.strip() or i == 0:
                        new_content.append({"type": "text", "text": part})
                if new_content and new_content[-1].get("type") == "text" and not new_content[-1].get("text", "").strip():
                    new_content.pop()
                formatted_messages.append({"role": msg["role"], "content": new_content})
            else:
                formatted_messages.append(msg)
        return formatted_messages

    def _prepare_inputs(self, messages: List[Dict[str, str]], images: List[str]):
        """Tokenize text + process images. Returns dict of tensors."""
        if process_vision_info is None:
            raise ImportError("qwen_vl_utils is required for Qwen3-VL training.")

        loaded_images = self._load_images(images)
        formatted_messages = self._format_messages(messages, loaded_images)

        text = self.processor.apply_chat_template(formatted_messages, tokenize=False, add_generation_prompt=False)
        image_inputs, video_inputs = process_vision_info(formatted_messages)

        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=False,
            return_tensors=None,  # we will pad in collator
            truncation=False,
        )
        # inputs is a dict of lists/tensors (batch size 1)
        inputs = {k: v[0] if isinstance(v, list) else v for k, v in inputs.items()}
        return inputs, text

    def _extract_labels(self, sample: Dict[str, Any], messages: List[Dict[str, str]], text: str, input_ids: List[int], prompt_len: int) -> Dict[str, Any]:
        """Override in subclasses to return stage-specific label tensors."""
        return {}

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        sample = self.samples[idx]
        messages = self._build_messages(sample)
        images = sample.get("images", [])
        loaded_images = self._load_images(images)

        # Full inputs (prompt + assistant) — must use formatted messages with embedded images
        full_formatted = self._format_messages(messages, loaded_images)
        inputs, text = self._prepare_inputs_from_formatted(full_formatted)
        input_ids = inputs["input_ids"]

        # Prompt-only inputs to determine assistant start position
        prompt_messages = messages[:-1]
        prompt_formatted = self._format_messages(prompt_messages, loaded_images)
        prompt_text = self.processor.apply_chat_template(prompt_formatted, tokenize=False, add_generation_prompt=True)
        prompt_image_inputs, prompt_video_inputs = process_vision_info(prompt_formatted)
        prompt_inputs = self.processor(
            text=[prompt_text],
            images=prompt_image_inputs,
            videos=prompt_video_inputs,
            padding=False,
            return_tensors=None,
            truncation=False,
        )
        prompt_input_ids = prompt_inputs["input_ids"]
        if isinstance(prompt_input_ids, list) and len(prompt_input_ids) > 0 and isinstance(prompt_input_ids[0], list):
            prompt_input_ids = prompt_input_ids[0]
        prompt_len = len(prompt_input_ids)

        # Labels: mask prompt tokens as -100
        labels = list(input_ids)
        for i in range(min(prompt_len, len(labels))):
            labels[i] = -100

        item = {
            "input_ids": input_ids,
            "labels": labels,
        }
        # Add image tensors if present
        for key in ["pixel_values", "image_grid_thw", "pixel_values_videos", "video_grid_thw"]:
            if key in inputs:
                item[key] = inputs[key]

        # Stage-specific extras
        extras = self._extract_labels(sample, messages, text, input_ids, prompt_len)
        item.update(extras)
        return item

    def _prepare_inputs_from_formatted(self, formatted_messages: List[Dict[str, Any]]):
        """Tokenize already-formatted messages. Returns dict of tensors (single sample)."""
        text = self.processor.apply_chat_template(formatted_messages, tokenize=False, add_generation_prompt=False)
        image_inputs, video_inputs = process_vision_info(formatted_messages)
        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=False,
            return_tensors=None,
            truncation=False,
        )
        inputs = {k: v[0] if isinstance(v, list) else v for k, v in inputs.items()}
        return inputs, text


# ---------------------------------------------------------------------------
# Data Collator
# ---------------------------------------------------------------------------

class PGDTSDataCollator:
    """
    Collates variable-length sequences and stage-specific tensors.
    Compatible with transformers.Trainer.
    """

    def __init__(self, processor: AutoProcessor, pad_to_multiple_of: Optional[int] = None):
        self.processor = processor
        self.tokenizer = processor.tokenizer
        self.pad_to_multiple_of = pad_to_multiple_of

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        # Separate standard keys from extra keys
        standard_keys = {"input_ids", "labels", "pixel_values", "image_grid_thw",
                         "pixel_values_videos", "video_grid_thw"}
        extra_keys = set(features[0].keys()) - standard_keys

        # Pad input_ids and labels using tokenizer.pad (most reliable)
        batch = self.tokenizer.pad(
            {"input_ids": [f["input_ids"] for f in features],
             "labels": [f["labels"] for f in features]},
            padding=True,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors="pt",
        )
        # Ensure int64
        batch = {k: v.to(torch.int64) if k in ("input_ids", "labels") else v for k, v in batch.items()}

        # Pad / stack image tensors
        # NOTE: Qwen3-VL returns pixel_values as [num_patches, C, H, W] per image.
        # In a batch, num_patches differs across images, so we concatenate along
        # dim 0 and keep image_grid_thw to tell the model where each image ends.
        if "pixel_values" in features[0]:
            pixel_values = [f["pixel_values"] for f in features]
            tensors = []
            for p in pixel_values:
                if isinstance(p, np.ndarray):
                    tensors.append(torch.from_numpy(p))
                elif isinstance(p, torch.Tensor):
                    tensors.append(p)
                else:
                    tensors.append(torch.tensor(p, dtype=torch.float32))
            batch["pixel_values"] = torch.cat(tensors, dim=0)

        if "image_grid_thw" in features[0]:
            grid = [f["image_grid_thw"] for f in features]
            tensors = []
            for g in grid:
                if isinstance(g, np.ndarray):
                    tensors.append(torch.from_numpy(g))
                elif isinstance(g, torch.Tensor):
                    tensors.append(g)
                else:
                    tensors.append(torch.tensor(g, dtype=torch.int64))
            batch["image_grid_thw"] = torch.stack(tensors, dim=0)

        # Handle extras
        for key in extra_keys:
            values = [f[key] for f in features]
            first = values[0]
            if isinstance(first, (int, float)):
                batch[key] = torch.tensor(values)
            elif isinstance(first, list):
                # Pad lists to max length in batch
                max_len = max(len(v) for v in values)
                if isinstance(first[0], int):
                    padded = [v + [-1] * (max_len - len(v)) for v in values]
                    batch[key] = torch.tensor(padded, dtype=torch.int64)
                elif isinstance(first[0], float):
                    padded = [v + [0.0] * (max_len - len(v)) for v in values]
                    batch[key] = torch.tensor(padded, dtype=torch.float32)
                else:
                    batch[key] = values  # keep as list of objects
            elif isinstance(first, np.ndarray):
                if all(v.shape == first.shape for v in values):
                    batch[key] = torch.stack([torch.from_numpy(v) for v in values])
                else:
                    batch[key] = values
            else:
                batch[key] = values

        return batch


# ---------------------------------------------------------------------------
# Base Trainer
# ---------------------------------------------------------------------------

class PGDTSTrainerBase(SFTTrainer):
    """
    Base Trainer for PGDTS (built on TRL's SFTTrainer).
    Handles model + LoRA initialisation, dataset creation, and standard
    autoregressive loss. Subclasses override `compute_loss` for custom terms.
    """
    # Prevent TRL from interfering with our custom dataset preparation
    _dataset_kwargs = {"skip_prepare_dataset": True}

    def __init__(self, args: PGDTSArguments, **kwargs):
        self.pgdts_args = args
        self.system_prompt = load_system_prompt(args.system_prompt)

        # Load processor & model
        self.processor = AutoProcessor.from_pretrained(args.model_id_or_path, trust_remote_code=True)
        model = AutoModelForVision2Seq.from_pretrained(
            args.model_id_or_path,
            torch_dtype=torch.bfloat16 if args.bf16 else (torch.float16 if args.fp16 else torch.float32),
            trust_remote_code=True,
            device_map=None,  # let Trainer handle device placement
        )

        # Apply LoRA
        model = self._apply_lora(model, args)

        # If adapter_path provided, load existing LoRA weights
        if args.adapter_path and os.path.isdir(args.adapter_path):
            model = PeftModel.from_pretrained(model, args.adapter_path, is_trainable=True)
            print(f"[INFO] Loaded adapter from {args.adapter_path}")

        # Hook for subclasses to inject custom modules before Trainer init
        model = self._post_init_model(model)

        # TrainingArguments
        training_args = TrainingArguments(
            output_dir=args.output_dir,
            num_train_epochs=args.num_train_epochs,
            per_device_train_batch_size=args.per_device_train_batch_size,
            per_device_eval_batch_size=args.per_device_eval_batch_size,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            learning_rate=args.learning_rate,
            lr_scheduler_type=args.lr_scheduler_type,
            weight_decay=args.weight_decay,
            warmup_ratio=args.warmup_ratio,
            bf16=args.bf16,
            fp16=args.fp16,
            save_steps=args.save_steps,
            eval_steps=args.eval_steps,
            logging_steps=args.logging_steps,
            save_total_limit=args.save_total_limit,
            seed=args.seed,
            dataloader_num_workers=args.dataloader_num_workers,
            remove_unused_columns=args.remove_unused_columns,
            group_by_length=args.group_by_length,
            evaluation_strategy="steps" if args.val_jsonl else "no",
            save_strategy="steps",
            logging_strategy="steps",
            load_best_model_at_end=False,
            report_to=["tensorboard"],
        )

        # Datasets
        train_dataset = self._build_dataset(args.train_jsonl)
        eval_dataset = self._build_dataset(args.val_jsonl) if args.val_jsonl else None

        # Collator
        data_collator = PGDTSDataCollator(self.processor)

        super().__init__(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            data_collator=data_collator,
            tokenizer=self.processor.tokenizer,
            dataset_kwargs=self._dataset_kwargs,
            **kwargs,
        )

    def _apply_lora(self, model: nn.Module, args: PGDTSArguments) -> nn.Module:
        lora_config = LoraConfig(
            r=args.lora_rank,
            lora_alpha=args.lora_alpha,
            target_modules=args.lora_target_modules,
            lora_dropout=args.lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()
        return model

    def _post_init_model(self, model: nn.Module) -> nn.Module:
        """Hook for subclasses to add custom heads/modules before Trainer init."""
        return model

    def _build_dataset(self, jsonl_path: str) -> PGDTSDataset:
        return PGDTSDataset(
            jsonl_path=jsonl_path,
            processor=self.processor,
            system_prompt=self.system_prompt,
            max_length=self.pgdts_args.max_length,
        )

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        """Standard autoregressive loss (used by Stage 3; overridden by Stages 1 & 2)."""
        # `inputs` already contains `labels` from the collator; pass it through.
        outputs = model(**inputs, use_cache=False)
        loss = outputs.loss
        # NOTE: Do NOT divide by gradient_accumulation_steps here.
        # outputs.loss is already mean-reduced; TRL/Trainer handles scaling internally.
        return (loss, outputs) if return_outputs else loss
