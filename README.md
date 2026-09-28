# mt-training

Fine-tuning and evaluation of NLLB-200 for French → Mooré (Mossi) machine translation.

## Setup

```bash
uv sync
cp .env.example .env  # fill in credentials
```

Required env vars for S3/eval:

```shell
AWS_ACCESS_KEY_ID=
AWS_SECRET_ACCESS_KEY=
AWS_ENDPOINT_URL_S3=https://fly.storage.tigris.dev
AWS_REGION=auto
```

## Training

```bash
./scripts/train.sh
```

Trains `facebook/nllb-200-distilled-600M` on [`madoss/moore-web-parallel`](https://huggingface.co/datasets/madoss/moore-web-parallel)
(config `mos-fra`, pinned to `v1.0.0`) and pushes to the Hub.

Extra arguments are passed through to training, and override the script's
own values: `./scripts/train.sh --output_dir_root ./runs/`.

### On Modal

`scripts/modal_train.py` runs any script from `scripts/` on a Modal GPU
(default A100-80GB), with outputs on the `mt-training-outputs` volume.

```bash
# once (uses the workspace secrets huggingface-secret and wandb-secret)
uvx modal setup

# smoke test, then a detached full run
uvx modal run scripts/modal_train.py --script debug.sh
uvx modal run --detach scripts/modal_train.py --script train.sh

# get the CTranslate2 model
uvx modal volume get mt-training-outputs nllb-600m-FrMos-ct2 ./nllb-600m-FrMos-ct2
```

See the docstring of `scripts/modal_train.py` for resuming and choosing the GPU.

## Inference

```bash
# Single sentence
uv run python -m mt_training.inference "Bonjour le monde"

# Interactive
uv run python -m mt_training.inference
```

## Evaluation

```bash
# Against the default S3 reference set
uv run python -m mt_training.eval

# Against a HuggingFace dataset
uv run python -m mt_training.eval \
    --dataset madoss/moore-web-parallel \
    --src_field french \
    --ref_field moore \
    --split test

# Save predictions
uv run python -m mt_training.eval --output predictions.jsonl
uv run python -m mt_training.eval --output predictions.csv --output_format csv
```

Key options: `--model`, `--batch_size`, `--limit`, `--src_lang`, `--tgt_lang`.  
Config files are also supported: `uv run python -m mt_training.eval --config eval.yaml`.
