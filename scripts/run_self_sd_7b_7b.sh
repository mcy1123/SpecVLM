#!/bin/bash
# ============================================================
# Self-SD: Qwen2.5-VL-7B → Qwen2.5-VL-7B (同模型 Target + Draft)
# 2x A6000 (48GB), 128 frames, 25K visual tokens
# ============================================================
#
# 显存分析 (每卡 48GB):
#   - 每卡跑完整模型，无流水线并行，零 PCIe 开销
#   - Target GPU: 14GB 权重 + 2.2GB KV + 1.3GB ViT ≈ 20GB / 48GB
#   - Draft GPU:  14GB 权重 + 2.2GB KV + 1.3GB ViT ≈ 20GB / 48GB
#
# 论文参考值 (Self-SD Qwen2.5-VL-7B, A100):
#   SD-Tree τ: 4.38 | SpecVLM τ: 3.83
#   AR tok/s: ~10.41 | SpecVLM: 15.59 tok/s, 1.50x

set -euo pipefail

export PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True'
export HF_HOME="/home/mcy/local2/huggingface"
export HF_HUB_CACHE="${HF_HOME}/hub"

# ---- ViT per-frame SDPA ----
export SPECVLM_PER_FRAME_VIT=1

# ---- KVCache 容量 ----
export SPECVLM_MAX_CACHE_LEN=40960

# ---- 可调参数 ----
FRAME_NUM=128
EVAL_NUM=20              # 统计显著性
MIN_PIXELS=195000        # 匹配论文 ~196 tok/frame → ~25K visual tokens
MAX_PIXELS=320000
GPU_IDS="4,5"            # 2x A6000 空闲

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

SAVE_DIR="results/qwen2_5_vl_${TASK}_self_sd_128f_25k_2xA6000"

echo "=============================================="
echo "Self-SD 7B-7B | 2xA6000 | 128f | ~25K tokens"
echo "GPUs: ${GPU_IDS}"
echo "min_pixels=${MIN_PIXELS} max_pixels=${MAX_PIXELS}"
echo "SPECVLM_PER_FRAME_VIT=${SPECVLM_PER_FRAME_VIT}"
echo "EVAL_NUM=${EVAL_NUM}"
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
    --min_pixels $MIN_PIXELS \
    --max_pixels $MAX_PIXELS \
    --save_path "$SAVE_DIR"

echo "Done. Results: ${SAVE_DIR}/metric.jsonl"
