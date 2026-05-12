export PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True'
export HF_HOME="/home/mcy/local2/huggingface"
export HF_HUB_CACHE="${HF_HOME}/hub"
export TRANSFORMERS_CACHE="${HF_HOME}/transformers"

MODEL_TYPE="qwen2_5_vl"
BASE_MODEL_PATH="/home/mcy/local2/models/Qwen2.5-VL-7B-Instruct"
DRAFT_MODEL_PATH="/home/mcy/local2/models/Qwen2.5-VL-7B-Instruct"

TASK="VideoDetailCaption"
DATA_PATH="/home/mcy/local2/datasets/VideoDetailCaption"

EVAL_NUM=5
MAX_NEW_TOKENS=256
DATA_NUM=100
DROP_RATE=0.9
GPU_IDS="1,3"

FRAME_NUM=128

CUDA_VISIBLE_DEVICES=$GPU_IDS python inference.py \
    --model_type $MODEL_TYPE \
    --base_model_path $BASE_MODEL_PATH \
    --draft_model_path $DRAFT_MODEL_PATH \
    --data_path $DATA_PATH \
    --task $TASK \
    --frame_num $FRAME_NUM \
    --evaluation_num $EVAL_NUM \
    --max_new_tokens $MAX_NEW_TOKENS \
    --drop_rate $DROP_RATE \
    --data_num $DATA_NUM \
    --gpu_ids $GPU_IDS \
    --setting self \
    --percentage 0.5 \
    --save_path "results/${MODEL_TYPE}_${TASK}_self_sd_drop_rate_${DROP_RATE}"
