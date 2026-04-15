#!/usr/bin/env sh
uv run python -m mt_training.train \
    --num_train_epochs 5 \
    --per_device_train_batch_size 32 \
    --per_device_eval_batch_size 64 \
    --gradient_accumulation_steps 2 \
    --eval_accumulation_steps 1 \
    --learning_rate 2e-5 \
    --lr_scheduler_type cosine_with_min_lr \
    --lr_scheduler_kwargs '{"min_lr": 1e-6}' \
    --eval_strategy steps \
    --eval_steps 100 \
    --save_strategy epoch \
    --predict_with_generate \
    --gradient_checkpointing \
    --push_to_hub \
    --hub_private_repo true \
    --hub_strategy end \
    --report_to trackio
