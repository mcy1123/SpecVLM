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
CALIBRATION_DIR="$RESULT_ROOT/calibration"

"$PYTHON_BIN" vista_inference.py \
    --base_model_path "$TARGET_MODEL_PATH" \
    --draft_model_path "$DRAFT_MODEL_PATH" \
    --data_path "$VIDEO_DATA_PATH" \
    --frame_num "${FRAME_NUM:-128}" \
    --evaluation_num "${CALIBRATION_NUM:-20}" \
    --sample_offset 0 \
    --data_num "${DATA_NUM:-100}" \
    --max_new_tokens "${MAX_NEW_TOKENS:-256}" \
    --gpu_ids "${GPU_IDS:-0,1,2,3}" \
    --target_gpu_ids 0,1,2 \
    --draft_gpu_ids 3 \
    --route_mode gated \
    --anchor_mode static \
    --gate_margin_threshold 0.0 \
    --skip_ar \
    --save_path "$CALIBRATION_DIR"

"$PYTHON_BIN" scripts/calibrate_gate.py \
    "$CALIBRATION_DIR/results.jsonl" --percentile 30 --value-only \
    > "$CALIBRATION_DIR/gate_threshold.txt"

echo "Calibrated threshold: $(cat "$CALIBRATION_DIR/gate_threshold.txt")"
echo "Export it with: export GATE_MARGIN_THRESHOLD=$(cat "$CALIBRATION_DIR/gate_threshold.txt")"
