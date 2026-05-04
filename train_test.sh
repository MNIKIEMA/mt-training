#!/usr/bin/env sh
RUNNER=""
if [ "${USE_UV:-0}" = "1" ]; then
    RUNNER="uv run"
fi
${RUNNER} python -m mt_training.train \
    --num_train_epochs 1 \
    --per_device_train_batch_size 2 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 1 \
    --eval_accumulation_steps 1 \
    --learning_rate 8e-5 \
    --eval_strategy steps \
    --eval_steps 10 \
    --save_strategy no \
    --max_train_samples 64 \
    --eval_subset_size 10 \
    --train_sampling_strategy group_by_length \
    --predict_with_generate \
    --report_to none \
    --max_steps 20 \
    --run_name test-infra-dry-run \
    --repo_name mixed-nllb-top200k-mt \
    --output_dir_root /tmp/ \
    --dataset_id madoss/fr-mos-final-data
