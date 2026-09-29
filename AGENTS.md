# AGENTS.md

This repo fine-tunes and evaluates NLLB-200 for French -> Moore translation.

Work like a small, careful research engineer. Keep changes simple. Prefer a boring
working training run over a clever abstraction.

## The Shape Of The Repo

- `src/mt_training/train.py`: training entrypoint. Uses Hugging Face `Seq2SeqTrainer`.
- `src/mt_training/text.py`: `normalize_for_nllb`, applied to every text before tokenization (training and inference); NLLB has no ’ or « ».
- `src/mt_training/eval.py`: dataset loading, translation, BLEU, and chrF++ evaluation.
- `src/mt_training/overlap.py`: finds benchmark sources that also appear in the training data (normalized exact match).
- `src/mt_training/compare.py`: compares two `eval.py --output` files per source-length bucket, with a paired bootstrap CI.
- `src/mt_training/inference.py`: HF and CTranslate2 inference helpers.
- `src/mt_training/convert_ct2.py`: converts a HF seq2seq checkpoint to CTranslate2.
- `src/mt_training/round_trip.py`: French round trip through English (quickmt) to paraphrase the French side; not backtranslation.
- `scripts/train.sh`: default full training run.
- `scripts/train_mixed_nllb_200k.sh`: mixed/top200k synthetic data training run.
- `scripts/train_round_trip.sh`: training on authentic + round-tripped French (Hub names still say "backtranslated").
- `scripts/debug.sh`: small dry-run training script.
- `src/mt_training/backtranslate.py`: Mooré monolingual sentences -> synthetic French pairs (moore-web-parallel schema, marked synthetic, `drop_reason` per pair); `scripts/backtranslate.sh` runs it (also on Modal).
- `scripts/train_mos_fra.sh`: Mooré → French with `train.sh`'s hyperparameters (backtranslation model).
- `scripts/modal_train.py`: runs any of the scripts above on a Modal GPU, outputs on a Modal Volume.

## Commands

Use `uv` unless there is a good reason not to.

```bash
uv sync
uv run python -m mt_training.train --help
uv run python -m mt_training.eval --help
uv run python -m mt_training.inference "Bonjour le monde"
```

Fast checks:

```bash
python3 -m py_compile src/mt_training/train.py src/mt_training/eval.py src/mt_training/inference.py
sh -n scripts/*.sh
```

Project checks, when available:

```bash
just lint
just format
just test
just typecheck
```

## Training Contract

Training is expensive. Do not casually run full training.

Use `scripts/debug.sh` for smoke tests. Full scripts are intended for GPU/RunPod-style
environments and push results to the Hub.

The trainer should:

- keep the W&B run open through post-training eval;
- evaluate the in-domain `test` split when present;
- convert the trained model to CTranslate2 for fast external evaluation;
- evaluate FLORES+ `devtest`;
- log metrics to W&B;
- save metrics locally;
- push the final HF model to the Hub.

Default external eval dataset:

```python
FLORES_PLUS = "openlanguagedata/flores_plus"
FLORES_DEFAULT_SPLIT = "devtest"
```

## Metrics

Keep metric names stable. W&B grouping matters.

- In-domain trainer test metrics: `test/...`
- FLORES+ CT2 metrics: `flores_plus/...`

Primary metrics are BLEU and chrF++. chrF++ is especially important for model
selection and low-resource MT signal.

## Data Assumptions

Training datasets are expected to contain:

- `source`
- `french`
- `moore`

The training code renames:

- `source` -> `data_source`
- `--source_field` (default `french`) -> `source`
- `--target_field` (default `moore`) -> `target`

Mooré → French swaps the two fields and the NLLB codes (`--src_lang
mos_Latn --tgt_lang fra_Latn`); see `scripts/train_mos_fra.sh`.

Do not silently change these column contracts. If a dataset has a different
schema, make the mapping explicit.

FLORES+ may need `HF_TOKEN`.

## Style

- Python target: 3.12.
- Line length: 100.
- First-party imports: `mt_training`.
- Prefer typed dataclass config fields for CLI arguments.
- Keep shell scripts POSIX `sh`.
- Keep generated model artifacts and experiment outputs out of commits unless
  explicitly requested.

## Editing Rules

- Make the smallest change that preserves the intended experiment.
- Do not delete or rewrite experiment notes unless asked.
- Do not change training hyperparameters casually. Script changes can alter
  expensive runs.
- If touching post-training evaluation, preserve the comparison between
  in-domain `test` and out-of-domain FLORES+.
- If touching CT2, verify both conversion and inference call sites.
- If touching W&B logging, keep `wandb.finish()` after post-training eval and
  Hub push.

## Logbooks

`.agents/logbooks/` records findings and decisions per module (training,
evaluation/inference, synthetic data) and the scores of full runs
(`experiments.md`). Read the relevant logbook before changing a module, and
add a dated entry (newest on top) when a run or a fix teaches something.
Format and index: `.agents/logbooks/README.md`.

## Git Hygiene

The repo often has untracked experiment files. Do not sweep them into commits.

Before committing:

```bash
git status --short
git diff --cached --stat
```

Commit only the files requested or the files clearly required by the task.

## What Good Looks Like

A good change here is easy to run, easy to compare, and easy to recover from.
It leaves behind clear metrics, stable scripts, and no mystery state.
