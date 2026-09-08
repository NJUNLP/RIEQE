#!/bin/bash

set -x

MODEL_PATH=/anonymous/path/model/Qwen3-4B-Thinking-2507

DATASET=subtask_nonCOT-unit-tgt-3mixed-new-scale

EPOCH=3
OUTPUT_DIR=/anonymous/path/model/lora/Qwen3-4B-t2507-$DATASET-${EPOCH}ep

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export NCCL_P2P_DISABLE="1"
export NCCL_IB_DISABLE="1"
llamafactory-cli train \
    --model_name_or_path ${MODEL_PATH} \
    --trust_remote_code \
    --stage sft \
    --do_train \
    --finetuning_type lora \
    --lora_rank 64 \
    --lora_target all \
    --dataset $DATASET \
    --template qwen3 \
    --cutoff_len 2000 \
    --overwrite_cache \
    --preprocessing_num_workers 16 \
    --dataloader_num_workers 4 \
    --output_dir ${OUTPUT_DIR} \
    --logging_steps 1 \
    --save_strategy epoch \
    --plot_loss \
    --overwrite_output_dir \
    --save_only_model false \
    --report_to none \
    --per_device_train_batch_size 8 \
    --gradient_accumulation_steps 1 \
    --learning_rate 1e-4 \
    --num_train_epochs $EPOCH \
    --lr_scheduler_type cosine \
    --warmup_ratio 0.05 \
    --bf16 \
    --ddp_timeout 180000000 