# Data Directory

This directory is reserved for training and validation data. Consistent with Stage 1 and Stage 2, users should prepare or obtain datasets in the expected format described below.

## Source Images

The original first-person navigation frames used in our experiments come from the **WalkVLM** dataset:

- **Project page**: https://walkvlm2024.github.io/
- **Paper**: WalkVLM — A diverse walking awareness dataset with 12k video-manual annotation pairs from Europe and Asia, providing a fair benchmark for blind walking tasks.

To obtain the source images, please refer to the WalkVLM project page or contact the authors directly.

## Data Format

If you construct your own training data, each line in `*.jsonl` should follow the **multimodal conversation format** for Qwen3-VL:

```json
{
  "messages": [
    {"role": "system", "content": "Your system prompt here..."},
    {"role": "user", "content": "<image>\nScene Perception Data:\n- Scene Category: ...\n- Relevant Obstacles:\n  1. ...\n\nPlease provide concise and essential navigation prompts:"},
    {"role": "assistant", "content": "at 11 o'clock direction, there are pedestrians passing by, be careful to avoid."}
  ],
  "images": ["/absolute/path/to/image.jpg"]
}
```

- `messages`: OpenAI-style conversation turns.
- `images`: Absolute paths to the first-person navigation frames.
- `<image>`: Placeholder token processed by the Qwen3-VL `AutoProcessor`.
- **Structured Perception Data** (user message): Pre-computed scene category and obstacle list (with clock-face direction & distance in steps), simulating the output of the front-end perception pipeline (YOLOE + SAM 3 + Depth-Pro).
- **Decision Text** (assistant message): Concise, action-oriented navigation prompts for visually impaired users.

### Stage-Specific JSONL Fields

| Stage | Required Fields | Optional Fields |
|-------|----------------|-----------------|
| **1** (Object Perception) | `messages`, `images` | `object_list` — list of relevant object names; if absent, auto-parsed from assistant text |
| **2** (Spatial Understanding) | `messages`, `images` | `spatial_info` — list of `{distance_steps, distance_token_text}`; if absent, auto-parsed from assistant text |
| **3** (Decision Generation) | `messages`, `images` | — |

## Expected File Layout

After preparing your data, place files here:

```
data/
├── stage1_train.jsonl   # Stage 1 training data
├── stage1_val.jsonl     # Stage 1 validation data
├── stage2_train.jsonl   # Stage 2 training data
├── stage2_val.jsonl     # Stage 2 validation data
├── train.jsonl          # Stage 3 training data
└── val.jsonl            # Stage 3 validation data
```

> System prompts can be included directly in the `messages` field of your JSONL (as a `"system"` turn), or supplied separately via the `--system_prompt` argument.

> ⚠️ **Image Path Notice**: Ensure the paths in the `images` field are valid on your system before training or inference.
