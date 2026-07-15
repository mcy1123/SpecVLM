#!/usr/bin/env bash
set -euo pipefail

: "${TARGET_MODEL_PATH:?Set TARGET_MODEL_PATH}"
: "${DRAFT_MODEL_PATH:?Set DRAFT_MODEL_PATH}"
: "${VIDEO_DATA_PATH:?Set VIDEO_DATA_PATH}"
: "${GATE_MARGIN_THRESHOLD:?Run scripts/run_calibration_h100.sh and export its threshold}"

export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export SPECVLM_PER_FRAME_VIT="${SPECVLM_PER_FRAME_VIT:-1}"
export SPECVLM_MAX_CACHE_LEN="${SPECVLM_MAX_CACHE_LEN:-40960}"

PYTHON_BIN="${PYTHON_BIN:-python}"
RESULT_ROOT="${RESULT_ROOT:-results/h100/ablation}"
GPU_IDS="${GPU_IDS:-0,1,2,3}"
FRAME_NUM="${FRAME_NUM:-128}"
EVAL_NUM="${EVAL_NUM:-20}"

run_variant() {
    local name="$1"
    local route="$2"
    local anchor="$3"
    "$PYTHON_BIN" vista_inference.py \
        --base_model_path "$TARGET_MODEL_PATH" \
        --draft_model_path "$DRAFT_MODEL_PATH" \
        --data_path "$VIDEO_DATA_PATH" \
        --frame_num "$FRAME_NUM" \
        --evaluation_num "$EVAL_NUM" \
        --sample_offset "${SAMPLE_OFFSET:-20}" \
        --data_num "${DATA_NUM:-100}" \
        --max_new_tokens "${MAX_NEW_TOKENS:-256}" \
        --gpu_ids "$GPU_IDS" \
        --target_gpu_ids 0,1,2 \
        --draft_gpu_ids 3 \
        --anchor_ratio "${ANCHOR_RATIO:-0.1}" \
        --anchor_ema "${ANCHOR_EMA:-0.8}" \
        --refresh_interval "${REFRESH_INTERVAL:-4}" \
        --gate_margin_threshold "$GATE_MARGIN_THRESHOLD" \
        --route_mode "$route" \
        --anchor_mode "$anchor" \
        --save_path "$RESULT_ROOT/$name"
}

run_variant implicit_only implicit_only static
run_variant static_anchor anchor_only static
run_variant dynamic_anchor anchor_only dynamic
run_variant vista gated dynamic
