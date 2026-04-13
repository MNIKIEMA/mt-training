# List available recipes
help:
    @just --list

# Format code [ruff]"
format:
    uvx ruff format src
# Run linters [ruff]"
lint:
    uvx ruff check src --fixable all

# Run pre-commit [lint, format]"
pre-commit: lint format
    uvx prek run

# Install dependencies [dev and cu128]"
install:
    uv sync --all-groups --extra cu128

# Install deps on runpod
runpod-install:
	uv pip install -e . --python /usr/bin/python3 --break-system-packages

# Clean cache files
clean:
    find . -type d -name __pycache__ -exec rm -rf {} +
    find . -type f -name "*.pyc" -delete
    rm -rf .pytest_cache .coverage htmlcov .ruff_cache

# Typecheck using ty
typecheck:
    uvx ty check

# Train NLLB model (override any arg via CLI, e.g. --num_train_epochs 5)
train:
    uv run python -m mt_training.train \
        --num_train_epochs 5 \
        --per_device_train_batch_size 16 \
        --per_device_eval_batch_size 16 \
        --gradient_accumulation_steps 8 \
        --eval_accumulation_steps 4 \
        --learning_rate 2e-5 \
        --lr_scheduler_type constant \
        --eval_strategy steps \
        --eval_steps 100 \
        --save_strategy epoch \
        --predict_with_generate \
        --gradient_checkpointing \
        --push_to_hub \
        --hub_private_repo \
        --hub_strategy end \
        --report_to trackio

# Run using Pytest
test:
	uv run pytest
