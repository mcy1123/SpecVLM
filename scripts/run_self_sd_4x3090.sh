#!/bin/bash
# ============================================================
# Self-SD Qwen2.5-VL-7B on 4x RTX 3090 (24GB)
# 匹配论文条件: 128 frames, ~25K visual tokens (~196 tok/frame)
# ============================================================
#
# 显存分析 (每卡 24GB):
#   - 模型权重:        ~7 GB  (两模型各一半层分布在 4 卡)
#   - KV Cache (36K):  ~0.8 GB
#   - ViT activations: ~3 GB  (仅 ViT 所在卡, per-frame SDPA)
#   - 峰值:            ~11 GB / 24 GB ✓
#
# 论文参考值 (Self-SD):
#   SD-Tree τ: 4.38 | SpecVLM τ: 3.83
#   AR tok/s: ~10.41 (A100)
#
# 注意: 3090 带宽 936 GB/s vs A100 2039 GB/s
#   Self-SD 加速比受限于带宽，τ 值不受影响

set -euo pipefail

export PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True'
export HF_HOME="/home/mcy/local2/huggingface"
export HF_HUB_CACHE="${HF_HOME}/hub"

# ---- ViT per-frame SDPA (消除 O(total²) mask, 25K tokens 必须) ----
export SPECVLM_PER_FRAME_VIT=1

# ---- KVCache 容量 (25K visual + text + 256 gen ≈ 36K, 留余量) ----
export SPECVLM_MAX_CACHE_LEN=40960

# ---- 可调参数 ----
FRAME_NUM=128
EVAL_NUM=5              # 快速验证用 5, 全量改 499
MIN_PIXELS=195000       # 匹配论文 ~196 tok/frame
MAX_PIXELS=320000
GPU_IDS="0,1,2,3"       # 4x 3090

MODEL_TYPE="qwen2_5_vl"
MODEL_PATH=/home/mcy/projects/models/Qwen2.5-VL-7B-Instruct
DATA_PATH=/home/mcy/projects/SpecVLM/datasets/VideoDetailCaption
TASK="VideoDetailCaption"
MAX_NEW_TOKENS=256
DATA_NUM=100
DROP_RATE=0.9
PERCENTAGE=0.5

SAVE_DIR="results/qwen2_5_vl_${TASK}_self_sd_128f_25k_4x3090"

echo "=============================================="
echo "Self-SD 7B-7B | 4x3090 | 128f | ~25K tokens"
echo "GPUs: ${GPU_IDS}"
echo "min_pixels=${MIN_PIXELS} max_pixels=${MAX_PIXELS}"
echo "SPECVLM_PER_FRAME_VIT=${SPECVLM_PER_FRAME_VIT}"
echo "SPECVLM_MAX_CACHE_LEN=${SPECVLM_MAX_CACHE_LEN}"
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
