# VISTA-SD MVP Implementation Plan

## Goal

Implement and validate a training-free, lossless VISTA-SD prototype on the
current 4x RTX 3090 server, then deliver portable scripts for formal throughput
measurements on a 4x H100 80GB server.

The local correctness pair is Qwen2.5-VL-7B (target) plus Qwen2.5-VL-3B
(draft). The H100 benchmark pair is Qwen2.5-VL-32B plus Qwen2.5-VL-7B.

The current server is used to establish correctness, stability, and complete
execution. H100 measurements determine the final speedup claim.

## Method

1. Run a full multimodal prefill for both target and draft models.
2. Keep the target dense for every verification pass.
3. During draft decoding, retain all textual/generated KV entries and select
   visual KV entries according to one of two routes:
   - **Implicit route:** read no visual KV during decoding; rely on the draft
     text KV states formed during multimodal prefill.
   - **Anchor route:** additionally read a small visual anchor set (10% by
     default).
4. Initialize the anchor bank from verifier text-to-visual prefill attention.
5. Refresh anchor scores with a lightweight verifier query/key side channel:

   `score_next = 0.8 * score_previous + 0.2 * normalized_verify_score`

6. Refresh every four verification rounds and immediately following a poor
   round (`accept_length <= 1`).
7. Route the next draft round using verifier confidence. The first round and a
   round following poor acceptance use anchors; otherwise the top-1/top-2
   verifier logit margin selects implicit versus anchor drafting.
8. Preserve exact target verification. VISTA-SD must produce the same greedy
   token sequence as target autoregressive decoding.

The MVP deliberately excludes model training, loose verification, adaptive
tree search, Sparrow weights, and image-only benchmarks.

## Implementation Stages

### Stage 1: reproducible baseline

- Reproduce AR, naive speculative decoding, and official SpecVLM with the local
  7B/3B model pair.
- Standardize warm-up, CUDA synchronization, segmented timing, JSONL metrics,
  model/data path configuration, and explicit target/draft device allocation.

### Stage 2: sparse draft visual KV

- Add Qwen2.5-VL draft-attention support for gathering selected visual KV while
  retaining all text/generated KV.
- Handle GQA, custom KVCache lengths, tree positions, and multi-GPU device maps.
- Require the all-visual selection to match dense draft logits within floating
  point tolerance.

### Stage 3: AnchorBank and routes

- Track visual-token ordinal-to-KV mappings, anchor indices, EMA scores,
  refresh count, and anchor-set Jaccard similarity.
- Fill the anchor budget using high verifier-attention tokens plus uniform
  temporal/spatial coverage.
- Expose implicit-only, anchor-only, static-anchor, and dynamic-anchor modes.

### Stage 4: verification-guided refresh and gate

- Collect reduced verifier QK relevance scores from the final four language
  layers without enabling full eager attention output.
- Select the accepted/rejected tree boundary after posterior evaluation and
  update the anchor bank.
- Calibrate the route-margin threshold on a fixed 20-sample calibration split.

### Stage 5: validation and H100 handoff

- Add unit and integration tests for selection, cache lengths, tree-boundary
  mapping, cleanup, exact output, and multi-sample execution.
- Validate 8-, 64-, and 128-frame runs locally.
- Provide environment preflight, smoke, baseline, ablation, benchmark, and
  result-summary scripts for 4x H100.

## Interfaces

The inference entry point will support:

- `--method vista_sd`
- `--anchor_ratio 0.1`
- `--anchor_ema 0.8`
- `--refresh_interval 4`
- `--anchor_attn_layers 4`
- `--gate_margin_threshold`
- `--failure_accept_threshold 1`
- `--route_mode implicit_only|anchor_only|gated`
- `--anchor_mode static|dynamic`
- `--target_gpu_ids`
- `--draft_gpu_ids`

Per-sample metrics include prefill/draft/verification/refresh time, decode
throughput, acceptance length, route counts, anchor refresh count/Jaccard,
visual KV budget, peak memory, and exact match against AR.

## Acceptance Criteria

Local completion requires:

- All methods finish without unhandled exceptions, NaNs, cache corruption, or
  device mismatches.
- All-visual sparse draft agrees with dense draft logits.
- Anchor selection has the correct budget and valid, unique indices.
- Both implicit and anchor routes execute in a gated run.
- Dynamic refresh executes and produces timing and anchor-change metrics.
- VISTA-SD matches target greedy AR token-for-token on the fixed correctness
  set.
- Repeated samples do not leak state or show monotonically increasing memory.

H100 performance is evaluated separately. The target outcome is at least 5%
decode-throughput improvement over the best comparable speculative baseline,
with 100% exact output matching. Negative or neutral results are retained and
reported rather than hidden.

## H100 Benchmark Configuration

- 4x H100 80GB.
- Qwen2.5-VL-32B target on GPUs 0,1,2 and Qwen2.5-VL-7B draft on GPU 3 for the
  128-frame preset.
- Paths supplied through `TARGET_MODEL_PATH`, `DRAFT_MODEL_PATH`,
  `VIDEO_DATA_PATH`, `RESULT_ROOT`, and `HF_HOME`.
- Primary runs: 64 and 128 frames, 256 generated tokens, at least 50 test
  samples, fixed inputs and device mapping across methods.
- Report decode and end-to-end speedup, mean/median/std, bootstrap 95% CI,
  acceptance length, refresh overhead, route ratio, memory, and per-sample
  results.
