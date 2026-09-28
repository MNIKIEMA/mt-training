# Evaluation and inference (`eval.py`, `inference.py`, `convert_ct2.py`)

`eval.py` translates a dataset and scores BLEU and chrF++ (FLORES+ `devtest`
by default, or any Hub dataset with `--split`). `inference.py` translates
with an HF model or a CTranslate2 model directory (`CT2Translator`).
Post-training evals in `train.py` go through CT2.

## 2026-09-28

- **HF inference ignored `--src_lang`.** `translate_batch` passed `src_lang`
  to the tokenizer call, which NLLB ignores, and `load_model` builds the
  tokenizer with `fra_Latn`: any other source language was still encoded as
  `fra_Latn`. It now sets `tokenizer.src_lang` first, as the CT2 path always
  did. French → Mooré evals were unaffected.
- **CT2 conversion needs its own RAM**: `TransformersConverter` loads the
  model on CPU. On a 7.5 GB laptop it can't run in the same process as
  training; standalone it works (int8, 10-example FLORES+ smoke run: chrF++
  28.6).

## 2026-04-17

- **Evaluate on both FLORES+ and in-domain data, and per source.**
  Fine-tuning moved FLORES+ from 22.4 to 23.2 chrF while the in-domain
  reference set reached 29.6; `max_new_tokens` 128 → 256 changed nothing.
  Scores and caveats in [experiments.md](experiments.md).
