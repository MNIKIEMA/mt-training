#!/usr/bin/env sh
RUNNER=""
if [ "${USE_UV:-0}" = "1" ]; then
    RUNNER="uv run"
fi
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True ${RUNNER} python -m mt_training.train \
    --num_train_epochs 8 \
    --per_device_train_batch_size 64 \
    --per_device_eval_batch_size 128 \
    --gradient_accumulation_steps 2 \
    --eval_accumulation_steps 1 \
    --learning_rate 1e-4 \
    --lr_scheduler_type cosine_with_min_lr \
    --lr_scheduler_kwargs '{"min_lr": 1e-6}' \
    --warmup_steps 0.1 \
    --logging_steps 20 \
    --eval_strategy epoch \
    --validation_size 500 \
    --save_strategy epoch \
    --save_total_limit 3 \
    --train_sampling_strategy group_by_length \
    --predict_with_generate \
    --gradient_checkpointing \
    --push_to_hub \
    --hub_private_repo true \
    --hub_strategy end \
    --report_to wandb \
    --max_steps -1 \
    --run_name mixed-nllb-top200k-result \
    --repo_name mixed-nllb-top200k-mt \
    --output_dir_root /workspace/ \
    --dataset_id madoss/fr-mos-final-data-nllb-top200k-dedup \
    --project nllb-moore-web
