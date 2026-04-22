"""
PGDTS Stage 1 Trainer — Scene Object Perception
================================================
Implements the Global Bag-of-Tokens Loss (L_Bag) as described in the paper:

  L_Stage1 = L_AR + λ₁ · L_Bag

where L_Bag is a time-dimension max-pooled, order-invariant multi-label
classification loss over the vocabulary.

Data format expected (multimodal JSONL, extended):
  {
    "messages": [...],
    "images": [...],
    "object_list": ["vehicle", "pedestrian", "bicycle"]   # optional; if absent,
                                                           # parsed from assistant text
  }
"""

import re
from typing import Any, Dict, List, Optional

import torch
import torch.nn.functional as F

from .pgdts_trainer_base import PGDTSDataset, PGDTSTrainerBase


class Stage1Dataset(PGDTSDataset):
    """Dataset that extracts object sets and builds vocabulary multi-hot vectors."""

    def _parse_object_list(self, sample: Dict[str, Any]) -> List[str]:
        """Parse the list of relevant objects from sample."""
        # Prefer explicit field
        if "object_list" in sample:
            return [str(o).strip().lower() for o in sample["object_list"]]

        # Fallback: parse from assistant text
        messages = sample.get("messages", [])
        if not messages:
            return []
        assistant_text = messages[-1].get("content", "")
        match = re.search(r"Relevant Objects:\s*(.+)", assistant_text, re.IGNORECASE | re.DOTALL)
        if match:
            objects = [o.strip().lower() for o in match.group(1).split(",") if o.strip()]
            return objects
        return []

    def _extract_labels(self, sample: Dict[str, Any], messages: List[Dict[str, str]], text: str, input_ids: List[int], prompt_len: int) -> Dict[str, Any]:
        objects = self._parse_object_list(sample)
        vocab_size = len(self.tokenizer)
        multi_hot = torch.zeros(vocab_size, dtype=torch.float32)

        for obj in objects:
            # Tokenize object name without special tokens
            token_ids = self.tokenizer.encode(obj, add_special_tokens=False)
            for tid in token_ids:
                if 0 <= tid < vocab_size:
                    multi_hot[tid] = 1.0

        return {"object_multi_hot": multi_hot.numpy()}


class Stage1Trainer(PGDTSTrainerBase):
    """
    Stage 1 trainer with Bag-of-Tokens auxiliary loss.
    """

    def _build_dataset(self, jsonl_path: str):
        return Stage1Dataset(
            jsonl_path=jsonl_path,
            processor=self.processor,
            system_prompt=self.system_prompt,
            max_length=self.pgdts_args.max_length,
        )

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        object_multi_hot = inputs.pop("object_multi_hot", None)  # [B, V]

        # Forward pass (labels are already inside `inputs` from the collator)
        outputs = model(**inputs, use_cache=False)
        ar_loss = outputs.loss

        # -------------------------------------------------------------------
        # Global Bag-of-Tokens Loss
        # -------------------------------------------------------------------
        if object_multi_hot is not None and self.pgdts_args.lambda_bag > 0:
            logits = outputs.logits  # [B, T, V]
            # Softmax over vocabulary dimension
            probs = F.softmax(logits, dim=-1)  # [B, T, V]
            # Max-pool over time dimension (Eq. 4.1)
            max_probs, _ = probs.max(dim=1)  # [B, V]

            # Move to same device
            object_multi_hot = object_multi_hot.to(max_probs.device)
            eps = 1e-7

            # Eq. 4.3: weighted binary cross-entropy
            alpha = self.pgdts_args.alpha_bag
            beta = self.pgdts_args.beta_bag

            pos_term = alpha * object_multi_hot * torch.log(max_probs + eps)
            neg_term = beta * (1.0 - object_multi_hot) * torch.log(1.0 - max_probs + eps)
            bag_loss = -(pos_term + neg_term).sum(dim=-1).mean()

            loss = ar_loss + self.pgdts_args.lambda_bag * bag_loss
        else:
            loss = ar_loss

        return (loss, outputs) if return_outputs else loss
