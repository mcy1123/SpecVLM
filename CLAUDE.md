# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

SpecVLM (EMNLP 2025) is a training-free speculative decoding framework that accelerates Video LLMs via **verifier-guided token pruning**. The draft model generates candidates on heavily pruned video tokens (up to 90%), while the target model verifies them in parallel using tree attention. Achieves 2.68x speedup on LLaVA-OneVision-72B and 2.11x on Qwen2.5-VL-32B.

Core insight: video draft model speculation is insensitive to visual token pruning, enabling aggressive token reduction without accuracy loss.

## Quick Commands

```bash
conda activate specvlm

# LLaVA-OneVision evaluation
sh run_llava.sh

# Qwen2.5-VL evaluation
sh run_qwenvl.sh
```

Edit the `.sh` files to configure model/dataset paths, pruning ratio (`DROP_RATE`), frame count (`FRAME_NUM`), and GPU IDs before running.

## Architecture

### Data Flow

```
Video → clip_input_video() → [video tokens | text tokens]
    → Two-stage prefill: video tokens → text tokens (with attention output)
    → Stage I pruning: attention-guided top-k selection from verifier
    → Stage II pruning: uniform spatial sampling on remaining tokens
    → Draft model generates candidate tree on pruned input
    → Target model parallel-verifies all candidates via tree attention
    → Accept matched tokens, reject mismatches, continue loop
```

### Method Modes

`inference.py` evaluates each sample under 3 modes for comparison:
- **AR (Autoregressive)**: standard greedy decoding (baseline)
- **Naive SD**: speculative decoding without video token pruning
- **SpecVLM**: speculative decoding with verifier-guided two-stage pruning

### Key Modules

| Module | File | Purpose |
|--------|------|---------|
| Entry point | `inference.py` | CLI parsing, model loading, evaluation loop with 3-mode comparison |
| Decoding engine (LLaVA) | `decoding/tree_decoding.py` | SD loop: prefill → draft → verify → update KV caches |
| Decoding engine (Qwen) | `decoding/tree_decoding_qwen2_5.py` | Same logic, Qwen-specific model API differences |
| Core utilities | `utils/utils.py` | `load_model()`, `load_data()`, `clip_input_video()`, all pruning functions, attention scoring |
| KV Cache | `kv_cache/kv_cache.py` | `KVCache` class, multi-GPU aware `initialize_past_key_values()` |
| Tree topology | `tree_choices/choices.py` | `mc_sim_7b_63` (24-node tree, 5 depths) and `chain` |
| Draft tree buffers | `utils/utils_c.py` | `Tree`/`node` classes, `generate_tree_buffers_draft()` for draft model tree attention |
| LLaVA model | `models/modeling_llava_onevision_tree.py` | Modified LLaVA-OneVision with tree attention and output hidden states |
| Qwen model | `models/modeling_qwen2_5_vl.py` | Modified Qwen2.5-VL with tree attention, returns `output_embeddings` and `video_hidden_states` |
| Qwen2 backbone | `models/modeling_qwen2_tree.py` | Qwen2 with tree attention mask support |

### Two-Stage Prefill Design

All decoding paths split input at the **last video token position**:
1. **Stage 1**: Prefill video tokens (extract visual features)
2. **Stage 2**: Prefill text tokens with `output_attentions=True` to capture cross-modal attention

This split is critical: attention scores from Stage 2 drive the verifier-guided pruning in SpecVLM.

### Pruning Functions (in `utils/utils.py`)

| Function | Strategy |
|----------|----------|
| `drop_visual_tokens_random()` | Random selection |
| `drop_visual_tokens_window()` | Keep contiguous window (front/middle/back) |
| `drop_visual_tokens_uniform()` | Uniform interval sampling |
| `drop_visual_tokens_by_attention()` | Top-K by verifier attention scores |
| `drop_visual_tokens_specvlm()` | **Combined**: attention-guided (Stage I) + uniform on remaining (Stage II) |

SpecVLM uses `percentage=0.4` by default: tokens contributing top 40% attention mass are kept by Stage I, remaining keep budget filled via uniform sampling (Stage II).

### Tree Decoding Pipeline

1. `generate_tree_buffers()` — Builds attention mask, position IDs, retrieval indices for target model tree attention
2. `generate_tree_buffers_draft()` — Builds attention mask list for draft model's level-by-level tree decoding
3. `tree_draft()` — Draft model autoregressively generates candidates level by level (TOPK=10 per node)
4. `generate_candidates()` — Flattens tree tokens into candidate paths
5. `tree_decoding()` — Target model verifies all candidates in one parallel forward pass
6. `evaluate_posterior()` — Greedy matching: accept longest consecutive match, reject mismatched suffix
7. `update_inference_inputs()` — Updates both target and draft KV caches from accepted path

### Model-Specific Differences

- **LLaVA-OV**: uses `video_token_id=151647`, `language_model.model.tree_mask`, `device_map="auto"`
- **Qwen2.5-VL**: uses `video_token_id=151656`, `model.tree_mask` (no language_model layer), `attn_implementation="sdpa"`, custom `Qwen2_5_VLProcessor`, fps-based frame control
- Qwen decoding (`tree_decoding_qwen2_5.py`) accesses model internals as `model.model` instead of `model.language_model.model`

### Key Parameters

| Parameter | Default | Purpose |
|-----------|---------|---------|
| `drop_rate` | 0.9 | Pruning ratio (90% of video tokens removed) |
| `percentage` | 0.4 | Attention mass threshold for Stage I selection |
| `FRAME_NUM` | 64/128 | Number of frames (higher = longer input = higher acceleration) |
| `max_new_tokens` | 256 | Generation length limit |
| `TOPK` | 10 | Top-k candidates at each tree branching node |
| `DEPTH` | 5 | Max tree depth |

### Dependencies

`transformers==4.48.0`, `torch`, `datasets`, `accelerate`, `qwen-vl-utils==0.0.10`, `av==14.0.0`

## Important Notes

- Code is based on the EAGLE speculative decoding framework (`SafeAILab/EAGLE`)
- Results save as JSONL to `results/` directory: per-sample records + aggregated metrics
- Longer video sequences (more frames) yield higher acceleration ratios — bounded by GPU memory bandwidth
- The method is lossless in principle; minor differences come from parallel decoding settings
- Qwen2.5-VL controls input length via fps rather than explicit frame selection — use `FRAME_NUM` to adjust
