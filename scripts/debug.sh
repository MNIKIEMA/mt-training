#!/usr/bin/env sh
RUNNER=""
if [ "${USE_UV:-0}" = "1" ]; then
    RUNNER="uv run"
fi
# Smoke tests exercise the W&B code path without uploading a run.
# Override with WANDB_MODE=online ./scripts/debug.sh
export WANDB_MODE="${WANDB_MODE:-offline}"
${RUNNER} python -m mt_training.train \
    --num_train_epochs 1 \
    --per_device_train_batch_size 2 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 1 \
    --max_length 64 \
    --eval_accumulation_steps 1 \
    --learning_rate 8e-5 \
    --logging_steps 10 \
    --eval_strategy steps \
    --eval_steps 10 \
    --save_strategy no \
    --max_train_samples 64 \
    --validation_size 10 \
    --post_training_eval_limit 10 \
    --train_sampling_strategy group_by_length \
    --predict_with_generate \
    --bf16 \
    --tf32 true \
    --max_steps 200 \
    --run_name test-infra-dry-run \
    --repo_name nllb-dry-run \
    --output_dir_root /tmp/ \
    --dataset_id madoss/moore-web-parallel \
    --dataset_config mos-fra \
    --dataset_revision v1.0.0 \
    --report_to wandb \
    --project nllb-moore-web \
    "$@"
