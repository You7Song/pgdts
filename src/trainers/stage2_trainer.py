"""
PGDTS Stage 2 Trainer — Spatial Understanding
==============================================
Implements the Distance Regression Loss (L_dist) as described in the paper:

  L_Stage2 = L_AR + λ_dist · L_dist

where L_dist uses a lightweight token-level linear regression head on the
hidden state at each distance-numeric token position, with distance-aware
Huber weighting.

Data format expected (multimodal JSONL, extended):
  {
    "messages": [...],
    "images": [...],
    "spatial_info": [                                   # optional
      {"object": "vehicle", "direction": "12 o'clock",
       "distance_steps": 2.0, "distance_token_text": "2"}
    ]
  }

If "spatial_info" is absent, distances and token texts are parsed from the
assistant response automatically via regex.
"""

import math
import re
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .pgdts_trainer_base import PGDTSDataset, PGDTSTrainerBase


class Stage2Dataset(PGDTSDataset):
    """Dataset that extracts distance token positions and ground-truth distances."""

    def _parse_spatial_info(self, sample: Dict[str, Any]) -> List[Tuple[float, str]]:
        """
        Parse list of (distance_steps, distance_token_text) from sample.
        Returns list of tuples; empty list if none found.
        """
        if "spatial_info" in sample:
            info = sample["spatial_info"]
            result = []
            for item in info:
                dist = float(item.get("distance_steps", -1))
                token_text = str(item.get("distance_token_text", "")).strip()
                if dist >= 0 and token_text:
                    result.append((dist, token_text))
            return result

        # Fallback: regex-parse from assistant text
        messages = sample.get("messages", [])
        if not messages:
            return []
        assistant_text = messages[-1].get("content", "")
        # Match patterns like "2 steps", "4 steps", "1–3 steps", "0 steps"
        # Capture the numeric part before "steps"
        matches = re.finditer(r"([\d–]+(?:\s*-\s*\d+)?)\s+steps", assistant_text, re.IGNORECASE)
        result = []
        for m in matches:
            token_text = m.group(1).strip()
            # Handle ranges like "1–3" by taking the median or lower bound
            if "–" in token_text or "-" in token_text:
                parts = re.split(r"[–\-]", token_text)
                try:
                    lo = float(parts[0].strip())
                    hi = float(parts[1].strip())
                    dist = (lo + hi) / 2.0
                except ValueError:
                    continue
            else:
                try:
                    dist = float(token_text)
                except ValueError:
                    continue
            result.append((dist, token_text))
        return result

    def _extract_labels(self, sample: Dict[str, Any], messages: List[Dict[str, str]], text: str, input_ids: List[int], prompt_len: int) -> Dict[str, Any]:
        spatial = self._parse_spatial_info(sample)
        if not spatial:
            return {"distance_positions": [-1], "distance_targets": [0.0]}

        assistant_text = messages[-1].get("content", "")
        # Tokenize the entire assistant text to build a char-to-token map
        enc = self.tokenizer(assistant_text, add_special_tokens=False)
        assistant_ids = enc["input_ids"]

        positions = []
        targets = []
        consumed_end = 0  # track end of last match to handle duplicate values

        for dist_val, token_text in spatial:
            # Find the next occurrence of token_text after consumed_end
            char_offset = assistant_text.find(token_text, consumed_end)
            if char_offset == -1:
                continue
            consumed_end = char_offset + len(token_text)

            # Tokenize prefix up to the numeric text to get relative token position
            prefix_text = assistant_text[:char_offset]
            prefix_ids = self.tokenizer(prefix_text, add_special_tokens=False)["input_ids"]
            rel_pos = len(prefix_ids)
            abs_pos = prompt_len + rel_pos

            # Sanity check: the token at abs_pos should start with the numeric text
            if abs_pos < len(input_ids) and rel_pos < len(assistant_ids):
                token_id = assistant_ids[rel_pos]
                decoded = self.tokenizer.decode([token_id], skip_special_tokens=True).strip()
                # Allow slight mismatch due to tokenizer whitespace variations
                if token_text not in decoded and decoded not in token_text:
                    # Fallback: search nearby tokens
                    found = False
                    for offset in range(-2, 3):
                        check_pos = rel_pos + offset
                        if 0 <= check_pos < len(assistant_ids):
                            check_decoded = self.tokenizer.decode([assistant_ids[check_pos]], skip_special_tokens=True).strip()
                            if token_text in check_decoded or check_decoded in token_text:
                                abs_pos = prompt_len + check_pos
                                found = True
                                break
                    if not found:
                        continue
                positions.append(abs_pos)
                targets.append(float(dist_val))

        if not positions:
            return {"distance_positions": [-1], "distance_targets": [0.0]}

        return {
            "distance_positions": positions,
            "distance_targets": targets,
        }


class Stage2Trainer(PGDTSTrainerBase):
    """
    Stage 2 trainer with token-level Distance Regression auxiliary loss.
    """

    def _post_init_model(self, model: nn.Module) -> nn.Module:
        hidden_size = model.config.hidden_size
        self.distance_head = nn.Linear(hidden_size, 1)
        # Register as a submodule so Trainer optimizer sees it
        model.distance_head = self.distance_head
        if next(model.parameters()).device != torch.device("cpu"):
            self.distance_head = self.distance_head.to(next(model.parameters()).device)
        print(f"[INFO] Stage 2: Added distance regression head (d={hidden_size}+1 params)")
        return model

    def _build_dataset(self, jsonl_path: str):
        return Stage2Dataset(
            jsonl_path=jsonl_path,
            processor=self.processor,
            system_prompt=self.system_prompt,
            max_length=self.pgdts_args.max_length,
        )

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        # Extract extra fields (labels stay inside `inputs` for the model forward)
        distance_positions = inputs.pop("distance_positions", None)   # [B, max_objects] int64
        distance_targets = inputs.pop("distance_targets", None)       # [B, max_objects] float32

        # Forward with hidden states (labels are already inside `inputs`)
        outputs = model(**inputs, use_cache=False, output_hidden_states=True)
        ar_loss = outputs.loss

        # -------------------------------------------------------------------
        # Distance Regression Loss
        # -------------------------------------------------------------------
        if (distance_positions is not None and distance_targets is not None
                and self.pgdts_args.lambda_dist > 0):
            hidden_states = outputs.hidden_states[-1]  # [B, T, D]
            B, T, D = hidden_states.shape

            # Valid mask: positions >= 0
            valid_mask = (distance_positions >= 0)  # [B, K]
            if valid_mask.any():
                # Gather hidden states at distance token positions
                pos_clamped = distance_positions.clamp(0, T - 1).to(hidden_states.device)  # [B, K]
                batch_idx = torch.arange(B, device=hidden_states.device).unsqueeze(1).expand_as(pos_clamped)
                selected_hidden = hidden_states[batch_idx, pos_clamped, :]  # [B, K, D]

                # Ensure distance_head is on the same device as hidden_states
                if self.distance_head.weight.device != hidden_states.device:
                    self.distance_head = self.distance_head.to(hidden_states.device)

                # Predict distances
                pred_distances = self.distance_head(selected_hidden).squeeze(-1)  # [B, K]

                # Ground truth
                gt_distances = distance_targets.to(hidden_states.device)  # [B, K]

                # Distance-aware weight ω(d) = exp(-d / τ)  (Eq. 4.8)
                tau = self.pgdts_args.dist_temperature
                weights = torch.exp(-gt_distances / tau)
                weights = weights * valid_mask.float()

                # Huber loss (Eq. 4.9)
                delta = self.pgdts_args.huber_delta
                residual = pred_distances - gt_distances
                abs_r = residual.abs()
                quadratic = torch.where(abs_r <= delta, 0.5 * residual ** 2, delta * (abs_r - 0.5 * delta))
                huber_loss = (weights * quadratic).sum() / valid_mask.float().sum().clamp_min(1.0)

                loss = ar_loss + self.pgdts_args.lambda_dist * huber_loss
            else:
                loss = ar_loss
        else:
            loss = ar_loss

        return (loss, outputs) if return_outputs else loss
