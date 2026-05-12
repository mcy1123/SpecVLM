#!/bin/bash
# ============================================================
# Self-SD: Qwen2.5-VL-7B → Qwen2.5-VL-7B (同模型 Target + Draft)
# 验证 SpecVLM 剪枝贡献，与论文 Table 2 Self-SD τ ≈ 3.83 对齐
# ============================================================
#
# GPU 需求: 2x 48GB (A6000)
#
# 分辨率与视觉 token 数（FRAME_NUM=128 时）:
#   默认 (max_pixels=200704)           → ~14K visual tokens
#   高分辨率 (min=195K, max=320K)      → ~25K visual tokens (匹配论文 ~196 tok/f)
#   超高分辨率 (min=200K, max=350K)    → ~28K visual tokens
#
# 已知结果（2x A6000, 128f, 5样本, percentage=0.5）:
#   AR:       15.51s, 15.72 tok/s
#   Naive SD: 18.82s, 13.75 tok/s, accept_len=4.46
#   SpecVLM:  16.37s, 15.76 tok/s, accept_len=3.84

set -euo pipefail

export PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True'
export HF_HOME="/home/mcy/local2/huggingface"
export HF_HUB_CACHE="${HF_HOME}/hub"

# ---- 可调参数 ----
FRAME_NUM=128          # 64 / 96 / 128
EVAL_NUM=5
GPU_IDS="2,5"          # 2 张空闲 GPU

# 分辨率控制（留空使用默认 max_pixels=200704）
MIN_PIXELS=""           # 例: 195000
MAX_PIXELS=""           # 例: 320000

MODEL_TYPE="qwen2_5_vl"
MODEL_PATH="/home/mcy/local2/models/Qwen2.5-VL-7B-Instruct"
DATA_PATH="/home/mcy/local2/datasets/VideoDetailCaption"
TASK="VideoDetailCaption"
MAX_NEW_TOKENS=256
DATA_NUM=100
DROP_RATE=0.9
PERCENTAGE=0.5

PIXEL_ARGS=""
[ -n "$MIN_PIXELS" ] && PIXEL_ARGS="$PIXEL_ARGS --min_pixels $MIN_PIXELS"
[ -n "$MAX_PIXELS" ] && PIXEL_ARGS="$PIXEL_ARGS --max_pixels $MAX_PIXELS"

SAVE_DIR="results/qwen2_5_vl_${TASK}_self_sd_${FRAME_NUM}frames"

echo "=============================================="
echo "Self-SD 7B-7B | ${FRAME_NUM} frames | GPUs: ${GPU_IDS}"
echo "min_pixels=${MIN_PIXELS:-default} max_pixels=${MAX_PIXELS:-default}"
echo "=============================================="

CUDA_VISIBLE_DEVICES=$GPU_IDS python inference.py \
    --model_type $MODEL_TYPE \
    --base_model_path $MODEL_PATH \
    --draft_model_path $MODEL_PATH \
    --data_path $DATA_PATH \
    --task $TASK \
    --frame_num $FRAME_NUM \
    --evaluation_num $EVAL_NUM \
    --max_new_tokens $MAX_NEW_TOKENS \
    --drop_rate $DROP_RATE \
    --data_num $DATA_NUM \
    --gpu_ids $GPU_IDS \
    --setting self \
    --percentage $PERCENTAGE \
    --save_path "$SAVE_DIR" \
    $PIXEL_ARGS

echo "Done. Results: ${SAVE_DIR}/metric.jsonl"
