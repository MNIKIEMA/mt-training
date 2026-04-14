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
./train.sh
```

Trains `facebook/nllb-200-distilled-600M` on the `madoss/fr-mos-final-data` dataset and pushes to the Hub.

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
    --dataset madoss/fr-mos-final-data \
    --src_field french \
    --ref_field moore \
    --split test

# Save predictions
uv run python -m mt_training.eval --output predictions.jsonl
uv run python -m mt_training.eval --output predictions.csv --output_format csv
```

Key options: `--model`, `--batch_size`, `--limit`, `--src_lang`, `--tgt_lang`.  
Config files are also supported: `uv run python -m mt_training.eval --config eval.yaml`.
