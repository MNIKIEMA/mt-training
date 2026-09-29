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
uvx modal run --detach scripts/modal_train.py --script train.sh --no-wait

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

Bouquet (`facebook/bouquet`) is supported as a second external benchmark:
`--dataset facebook/bouquet` (test split, sentence level by default;
`--bouquet_level paragraph_level --max_new_tokens 384` for paragraphs).

### Comparing two models

```bash
# predictions of both models on the same set, then per-length comparison
uv run python -m mt_training.eval --model base-ct2 --output base.csv
uv run python -m mt_training.eval --model new-ct2 --output new.csv
uv run mt-training compare --baseline base.csv --candidate new.csv
```

Prints chrF++ and BLEU per source-length bucket (4 equal-size buckets by
default), the paired bootstrap 95% CI of the chrF++ gain, and each model's
output/reference length ratio.

### Checking a benchmark against the training data

```bash
uv run mt-training overlap --benchmark burkimbia/mt-benchmark-public --output overlap.csv
```

Lists benchmark rows whose French source also appears in the training dataset
(default `madoss/moore-web-parallel` `mos-fra` `v1.0.0`), after normalizing
case, punctuation and spacing, with where it was seen and its word count.
Exact matches only, so it is a lower bound; short matches are usually common
words, 6+ word matches point to shared documents.
