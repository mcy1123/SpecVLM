#!/usr/bin/env bash
set -euo pipefail

: "${TARGET_MODEL_PATH:?Set TARGET_MODEL_PATH}"
: "${DRAFT_MODEL_PATH:?Set DRAFT_MODEL_PATH}"
: "${VIDEO_DATA_PATH:?Set VIDEO_DATA_PATH}"

export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export SPECVLM_PER_FRAME_VIT="${SPECVLM_PER_FRAME_VIT:-1}"
export SPECVLM_MAX_CACHE_LEN="${SPECVLM_MAX_CACHE_LEN:-8192}"

PYTHON_BIN="${PYTHON_BIN:-python}"
RESULT_ROOT="${RESULT_ROOT:-results/h100}"
GPU_IDS="${GPU_IDS:-0,1,2,3}"

"$PYTHON_BIN" vista_inference.py \
    --base_model_path "$TARGET_MODEL_PATH" \
    --draft_model_path "$DRAFT_MODEL_PATH" \
    --data_path "$VIDEO_DATA_PATH" \
    --frame_num 8 \
    --evaluation_num 1 \
    --data_num 5 \
    --max_new_tokens 32 \
    --gpu_ids "$GPU_IDS" \
    --target_gpu_ids 0,1,2 \
    --draft_gpu_ids 3 \
    --route_mode gated \
    --anchor_mode dynamic \
    --refresh_interval 4 \
    --gate_margin_threshold "${GATE_MARGIN_THRESHOLD:-0.1}" \
    --save_path "$RESULT_ROOT/smoke"
