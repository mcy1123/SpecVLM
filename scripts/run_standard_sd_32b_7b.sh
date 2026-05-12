#!/bin/bash
# ============================================================
# Standard SD: Qwen2.5-VL-32B (Target) + Qwen2.5-VL-7B (Draft)
# SpecVLM 主实验，论文 Table 2 加速比 2.11x
# ============================================================
#
# GPU 需求（帧数 ↑ → GPU ↑）:
#   64 frames  → 2-3 GPU (Target + Draft 可挤在 2 卡)
#   96 frames  → 3 GPU
#   128 frames → 4 GPU
#
# 已知结果:
#   64f (2 GPU):  AR=36.72s, SD=24.90s, SpecVLM=25.14s, 加速比=1.47x
#   96f (3 GPU):  AR=43.04s, SD=29.02s, SpecVLM=28.77s, 加速比=1.51x
#   128f (4 GPU): AR=52.30s, SD=35.97s, SpecVLM=31.73s, 加速比=1.65x
#
# 分辨率控制: 设置 MIN_PIXELS / MAX_PIXELS 调整视觉 token 密度

set -euo pipefail

export PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True'
export HF_HOME="/home/mcy/local2/huggingface"
export HF_HUB_CACHE="${HF_HOME}/hub"

# ---- 可调参数 ----
FRAME_NUM=128          # 64 / 96 / 128
EVAL_NUM=5

# GPU 分配（当前空闲: 1,2,3,5）
#   FRAME_NUM=64  → GPU_IDS="2,3,5"
#   FRAME_NUM=96  → GPU_IDS="2,3,5"
#   FRAME_NUM=128 → GPU_IDS="1,2,3,5"
GPU_IDS="1,2,3,5"

# 分辨率控制（留空使用默认 max_pixels=200704）
MIN_PIXELS=""           # 例: 195000
MAX_PIXELS=""           # 例: 320000

MODEL_TYPE="qwen2_5_vl"
BASE_MODEL="/home/mcy/local2/models/Qwen2.5-VL-32B-Instruct"
DRAFT_MODEL="/home/mcy/local2/models/Qwen2.5-VL-7B-Instruct"
DATA_PATH="/home/mcy/local2/datasets/VideoDetailCaption"
TASK="VideoDetailCaption"
MAX_NEW_TOKENS=256
DATA_NUM=100
DROP_RATE=0.9
PERCENTAGE=0.4        # 论文默认 0.4

PIXEL_ARGS=""
[ -n "$MIN_PIXELS" ] && PIXEL_ARGS="$PIXEL_ARGS --min_pixels $MIN_PIXELS"
[ -n "$MAX_PIXELS" ] && PIXEL_ARGS="$PIXEL_ARGS --max_pixels $MAX_PIXELS"

SAVE_DIR="results/qwen2_5_vl_${TASK}_standard_sd_${FRAME_NUM}frames"

echo "=============================================="
echo "Standard SD 32B-7B | ${FRAME_NUM} frames | GPUs: ${GPU_IDS}"
echo "min_pixels=${MIN_PIXELS:-default} max_pixels=${MAX_PIXELS:-default}"
echo "=============================================="

CUDA_VISIBLE_DEVICES=$GPU_IDS python inference.py \
    --model_type $MODEL_TYPE \
    --base_model_path $BASE_MODEL \
    --draft_model_path $DRAFT_MODEL \
    --data_path $DATA_PATH \
    --task $TASK \
    --frame_num $FRAME_NUM \
    --evaluation_num $EVAL_NUM \
    --max_new_tokens $MAX_NEW_TOKENS \
    --drop_rate $DROP_RATE \
    --data_num $DATA_NUM \
    --gpu_ids $GPU_IDS \
    --setting standard \
    --percentage $PERCENTAGE \
    --save_path "$SAVE_DIR" \
    $PIXEL_ARGS

echo "Done. Results: ${SAVE_DIR}/metric.jsonl"
