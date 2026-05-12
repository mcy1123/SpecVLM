#!/bin/bash
# ============================================================
# 单卡配置: Qwen2.5-VL-7B
#   AR 模式: 任意帧数均可
#   Self-SD: 64 帧可行，128 帧可能 OOM（两份模型 + KV cache）
#   如需纯 AR 测量，改 SETTING="self" 并只看 AR 行即可
# ============================================================
#
# GPU 需求: 1x 48GB (A6000)
#
# 推荐配置:
#   快速验证:     FRAME_NUM=64, SETTING=self
#   纯 AR 基线:   FRAME_NUM=128, SETTING=self (只看 metric.jsonl 的 AR 行)
#   高分辨率 AR:  FRAME_NUM=128, MIN_PIXELS=195000, MAX_PIXELS=320000

set -euo pipefail

export PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True'
export HF_HOME="/home/mcy/local2/huggingface"
export HF_HUB_CACHE="${HF_HOME}/hub"

# ---- 可调参数 ----
FRAME_NUM=64           # 64 (稳妥) / 128 (AR only 可行)
EVAL_NUM=100
GPU_IDS="4"            # 单张空闲 GPU: 1, 2, 3, 5 均可
SETTING="self"         # self=同模型Target+Draft, 只看AR指标可用self

# 分辨率（留空=默认 ~7K visual@64f, ~14K@128f）
MIN_PIXELS=""
MAX_PIXELS=""

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

SAVE_DIR="results/qwen2_5_vl_${TASK}_1gpu_${FRAME_NUM}frames"

echo "=============================================="
echo "Single GPU | 7B | ${FRAME_NUM} frames | GPU: ${GPU_IDS}"
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
    --setting $SETTING \
    --percentage $PERCENTAGE \
    --save_path "$SAVE_DIR" \
    $PIXEL_ARGS

echo "Done. Results: ${SAVE_DIR}/metric.jsonl"
