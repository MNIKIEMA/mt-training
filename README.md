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
(config `mos-fra`, pinned to `v1.1.0`) and pushes to the Hub.

Mooré → French (for backtranslation), same hyperparameters:

```bash
./scripts/train_mos_fra.sh
```

It passes `--source_field moore --target_field french --src_lang mos_Latn
--tgt_lang fra_Latn`; any direction can be trained with these four options.

### Backtranslation

```bash
# Mooré sentences (madoss/moore-web-mono v1.1.0) -> synthetic French pairs
uv run mt-training backtranslate --model <Mooré -> French model or CT2 dir>
uvx modal run --detach scripts/modal_train.py --script backtranslate.sh --no-wait \
    --extra-args "--model madoss/nllb-600m-MosFr-mwp-v1"
```

Writes pairs in the `moore-web-parallel` schema (`original_lang: mos`,
`reviewed: false`, `source: bt-<source>`, the Mooré sentence's `id`) with a
`drop_reason` (null, `empty`, `copy`, `moore_letters`, `length_ratio`, `loop`);
reruns skip ids already written.

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
default), the paired bootstrap 95% CI of the chrF++ gain, and for each model
the output/reference length ratio, the share of outputs over 1.5× their
reference length (`--long_ratio`) and the mean share of repeated words.
chrF++ rewards recall, so a gain that comes with longer or looping output is
not only a better translation: read these columns and BLEU together.

### The standard evaluation suite

```bash
# FLORES+ devtest, Bouquet sentences and paragraphs, moore-web-parallel test
./scripts/eval_suite.sh <model or CT2 dir> v3 evaluations/
./scripts/eval_suite.sh <other model> bt-v1 evaluations/
./scripts/compare_suite.sh v3 bt-v1 evaluations/

# Mooré -> French
SRC_LANG=mos_Latn TGT_LANG=fra_Latn ./scripts/eval_suite.sh <mos-fra model> mf-v2 evaluations/
```

`eval_suite.sh` scores with references normalized for the NLLB vocabulary
(`--normalize_references true`) and the `--max_new_tokens` used in the
logbooks (384 for Bouquet paragraphs, 256 for the test split); extra eval
options go in `EVAL_ARGS`. `compare_suite.sh` prints the `All` row of
`mt-training compare` per set and saves the per-bucket rows as JSON.

### Checking a benchmark against the training data

```bash
uv run mt-training overlap --benchmark burkimbia/mt-benchmark-public --output overlap.csv
```

Lists benchmark rows whose French source also appears in the training dataset
(default `madoss/moore-web-parallel` `mos-fra` `v1.0.0`), after normalizing
case, punctuation and spacing, with where it was seen and its word count.
Exact matches only, so it is a lower bound; short matches are usually common
words, 6+ word matches point to shared documents.
