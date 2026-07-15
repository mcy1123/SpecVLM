# Local Validation Report

## Environment

- 4x NVIDIA RTX 3090 24GB
- PyTorch 2.6.0+cu124
- Transformers 4.48.0
- Target: Qwen2.5-VL-7B-Instruct
- Draft: Qwen2.5-VL-3B-Instruct
- Dataset: VideoDetailCaption

These checks validate correctness and portability. RTX 3090 timings are not
used to claim speedup; formal performance measurements are reserved for the
4x H100 configuration in `H100_RUNBOOK.md`.

## Completed Checks

| Check | Configuration | Result |
| --- | --- | --- |
| Official baseline | 8 frames, 1 sample, 8 tokens | AR, naive SD, and SpecVLM completed with identical output |
| Static anchors | 8 frames, 10% anchors | Completed; exact match with AR |
| Dense fallback | 8 frames, 100% anchors | Completed; exact match with AR |
| Dynamic + gated routes | 8 frames | Dynamic refresh completed; both implicit and anchor routes executed |
| Long visual input | 64 frames, 4,340 visual tokens | Completed; exact match with AR |
| Multi-GPU long input | 128 frames, 8,820 visual tokens | Completed on target GPUs 0,1 and draft GPUs 2,3; exact match with AR |
| Consecutive regression | 5 videos, 8 frames, 16 tokens | 5/5 exact matches; no state leakage or monotonic memory growth |
| Unit tests | AnchorBank, sparse gather, and runtime collector | 9/9 passed |

An additional two-sample, 64-frame, 32-token comparison was run on the local
7B/3B pair. VISTA remained exact on both samples, but did not outperform the
official SpecVLM baseline on RTX 3090. Precomputing sparse prefix indices reduced
the first sample's VISTA decode time from 6.38s to 5.79s (about 9%), while the
corresponding SpecVLM sample took 3.26s. This is a negative local performance
result and reinforces that H100 32B/7B measurements are required before making
any acceleration claim.

The 64-frame run used both routes and performed four anchor refreshes. The
128-frame run exercised four-GPU model placement and performed a verifier-
guided refresh without cache or device errors.

## Reproduction Commands

Unit tests:

```bash
python -m unittest discover -s tests -v
```

Representative local smoke configuration:

```bash
export SPECVLM_PER_FRAME_VIT=1
export SPECVLM_MAX_CACHE_LEN=16384
python vista_inference.py \
  --base_model_path /path/to/Qwen2.5-VL-7B-Instruct \
  --draft_model_path /path/to/Qwen2.5-VL-3B-Instruct \
  --data_path /path/to/VideoDetailCaption \
  --frame_num 64 --evaluation_num 1 --max_new_tokens 16 \
  --gpu_ids 0,1 --target_gpu_ids 0 --draft_gpu_ids 1 \
  --route_mode gated --anchor_mode dynamic \
  --refresh_interval 4 --gate_margin_threshold 0.1 \
  --save_path results/local_smoke
```

## Known Limitations

- The route threshold must be recalibrated for the 32B/7B H100 model pair.
- Full draft prefill and full draft KV storage optimize decode, not prefill
  memory or end-to-end latency.
- Current visual relevance uses language-layer QK scores, without OCR/object
  priors or a trained gate.
- Anchor sets remained stable in the small local regression, so H100 testing
  must determine whether dynamic refresh improves acceptance over static
  anchors on longer generations and more diverse videos.
- Sparse `index_select` still materializes compact K/V tensors in every draft
  layer. If H100 results remain slower than SpecVLM, the next optimization is a
  fused sparse-attention kernel or a persistent compact draft KV cache.
