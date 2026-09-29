# Evaluation and inference (`eval.py`, `inference.py`, `convert_ct2.py`)

`eval.py` translates a dataset and scores BLEU and chrF++ (FLORES+ `devtest`
by default, or any Hub dataset with `--split`). `inference.py` translates
with an HF model or a CTranslate2 model directory (`CT2Translator`).
Post-training evals in `train.py` go through CT2.

## 2026-09-29 (evaluation suite)

- **`scripts/eval_suite.sh` / `scripts/compare_suite.sh`** replace the
  ad-hoc driver used for v1 vs v2 and bt-v1 vs v3: the same four sets and
  settings (normalized references, `--max_new_tokens` 384 for Bouquet
  paragraphs, 256 for the test split), CSVs named `<name>-<set>.csv`.
- **`mt-training compare` flags length and loops**: share of outputs over
  1.5× their reference length and mean repeated-word share
  (1 − distinct/total words) per model. Added after FrMos-bt-v1's FLORES+
  chrF++ gain (+2.28) came with outputs 22% longer and BLEU down
  ([experiments.md](experiments.md)); it reproduces the numbers found by
  hand (long 6.0% → 18.9%, repeated words 0.135 → 0.214).

## 2026-09-29 (Bouquet)

- **`eval.py --dataset facebook/bouquet`** loads a Bouquet benchmark file
  (`benchmark_data/<level>/<split>/<src>-<tgt>.parquet`, default split
  `test`, `--bouquet_level sentence_level|paragraph_level`), keeping
  `domain`. fra-mos test: 854 sentences, 198 paragraphs. Paragraph
  references reach 271 Mooré tokens: raise `--max_new_tokens` (384 used) or
  outputs get cut at 128.

## 2026-09-28 (FLORES+ for any model)

- **`uvx modal run scripts/modal_train.py --flores-model <hub id>`** (was
  `::flores` until 2026-09-29; add `--src-lang mos_Latn --tgt-lang fra_Latn`
  for Mooré → French) runs
  the post-training FLORES+ eval on any HF model: CT2 conversion (int8) +
  `run_evaluation` with the same settings, on the same Modal GPU, so scores
  are comparable with post-training results. Translations are saved to the
  `mt-training-outputs` volume under `eval/`, for significance tests.
- **CT2 on GPU vs CPU**: `CT2Translator` uses `device="auto"`, so on a
  Modal GPU it should run on CUDA (not verified; the "Model loaded on: cpu"
  line is printed for any CT2 model, whatever the device). int8 kernels differ between CPU and GPU, so compare FLORES+ scores
  from the same device type.

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
