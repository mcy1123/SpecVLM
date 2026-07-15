#!/usr/bin/env bash
set -euo pipefail

: "${TARGET_MODEL_PATH:?Set TARGET_MODEL_PATH}"
: "${DRAFT_MODEL_PATH:?Set DRAFT_MODEL_PATH}"
: "${VIDEO_DATA_PATH:?Set VIDEO_DATA_PATH}"

export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export SPECVLM_PER_FRAME_VIT="${SPECVLM_PER_FRAME_VIT:-1}"
export SPECVLM_MAX_CACHE_LEN="${SPECVLM_MAX_CACHE_LEN:-40960}"

PYTHON_BIN="${PYTHON_BIN:-python}"
RESULT_ROOT="${RESULT_ROOT:-results/h100}"
GPU_IDS="${GPU_IDS:-0,1,2,3}"
FRAME_NUM="${FRAME_NUM:-128}"
EVAL_NUM="${EVAL_NUM:-50}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-256}"

"$PYTHON_BIN" inference.py \
    --model_type qwen2_5_vl \
    --base_model_path "$TARGET_MODEL_PATH" \
    --draft_model_path "$DRAFT_MODEL_PATH" \
    --data_path "$VIDEO_DATA_PATH" \
    --task VideoDetailCaption \
    --frame_num "$FRAME_NUM" \
    --evaluation_num "$EVAL_NUM" \
    --sample_offset "${SAMPLE_OFFSET:-20}" \
    --max_new_tokens "$MAX_NEW_TOKENS" \
    --drop_rate "${DROP_RATE:-0.9}" \
    --percentage "${PERCENTAGE:-0.5}" \
    --data_num "${DATA_NUM:-100}" \
    --gpu_ids "$GPU_IDS" \
    --target_gpu_ids 0,1,2 \
    --draft_gpu_ids 3 \
    --save_path "$RESULT_ROOT/baselines_${FRAME_NUM}f"
