#!/usr/bin/env bash
set -euo pipefail

export CUDA_HOME=/home/xxf/anaconda3/envs/distill92/lib/python3.12/site-packages/nvidia/cu13
export PATH=/home/xxf/anaconda3/envs/distill92/lib/python3.12/site-packages/nvidia/cu13/bin:$PATH

export PYTHONHASHSEED=42
FULL_DETERMINISM=${FULL_DETERMINISM:-true}
if [[ "$FULL_DETERMINISM" == "true" ]]; then
    export CUBLAS_WORKSPACE_CONFIG=:4096:8
    export FLASH_ATTENTION_DETERMINISTIC=1
    export CUDA_LAUNCH_BLOCKING=1
else
    unset CUBLAS_WORKSPACE_CONFIG FLASH_ATTENTION_DETERMINISTIC CUDA_LAUNCH_BLOCKING
fi

REPO_DIR=/home/xxf/Distill/trl
OUTPUT_DIR=/home/xxf/Distill/Qbitwise/Qwen3.5-0.8B-Base-MiniLLM
PYTHON=/home/xxf/anaconda3/envs/distill92/bin/python

mkdir -p "$OUTPUT_DIR"
cd "$REPO_DIR"

CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0} PYTHONUNBUFFERED=1 \
"$PYTHON" /home/xxf/Distill/trl/experiment/minillm.py \
    --model_name_or_path /home/xxf/models/Qwen3.5-0.8B-Base \
    --teacher_model_name_or_path /home/xxf/models/Qwen3.5-2B \
    --dataset_name /home/xxf/Distill/data-tldr \
    --output_dir "$OUTPUT_DIR" \
    --dtype bfloat16 \
    --bf16 true \
    --full_determinism "$FULL_DETERMINISM" \
    --use_vllm true \
    --use-vllm-teacher false \
    --teacher_vllm_gpu_memory_utilization 0.4 \
    --vllm_enable_sleep_mode true \
    --vllm_importance_sampling_correction false \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 8 \
    --num_generations 1 \
    --max_completion_length 128 \
    --learning_rate 5e-6 \
    --num_train_epochs 1 \
    --logging_steps 1 \
    --save_steps 200 \
    --report_to tensorboard \
    2>&1 | tee "$OUTPUT_DIR/trainer.log"
