# H100 Runbook

## Environment

Create the same Python environment described by `requirements.txt`, then set:

```bash
export TARGET_MODEL_PATH=/path/to/Qwen2.5-VL-32B-Instruct
export DRAFT_MODEL_PATH=/path/to/Qwen2.5-VL-7B-Instruct
export VIDEO_DATA_PATH=/path/to/VideoDetailCaption
export RESULT_ROOT=/path/to/results/h100
export HF_HOME=/path/to/huggingface-cache
```

The default H100 preset exposes GPUs `0,1,2,3`, places the target on logical
GPUs `0,1,2`, and the draft on logical GPU `3`.

Calibration uses shuffled samples 0–19. Formal benchmark scripts default to
`SAMPLE_OFFSET=20`, so calibration and test samples do not overlap.

## Execution

```bash
bash scripts/preflight_h100.sh
bash scripts/run_smoke_h100.sh
EVAL_NUM=20 bash scripts/run_calibration_h100.sh
export GATE_MARGIN_THRESHOLD=$(cat "$RESULT_ROOT/calibration/gate_threshold.txt")
bash scripts/run_benchmark_h100.sh
bash scripts/run_ablation_h100.sh
```

Override any experiment parameter with environment variables, for example:

```bash
EVAL_NUM=5 FRAME_NUM=64 MAX_NEW_TOKENS=64 bash scripts/run_vista_h100.sh
```

Do not compare timings unless all methods use the same model paths, input
samples, frame count, generation cap, GPU visibility, and device mapping.

The smoke run must report `exact_match_ar: true` before launching the full
benchmark. Results are stored as per-sample JSONL and aggregated with
`scripts/summarize_results.py`.
