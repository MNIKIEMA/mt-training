# Changelog

All notable changes to this project will be documented in this file.

## [Unreleased]

### Added

- `scripts/modal_train.py` runs a training script on a Modal GPU: image
  built from `uv.lock`, outputs and HF cache on Modal Volumes, `HF_TOKEN`
  and `WANDB_API_KEY` from the `huggingface-secret` and `wandb-secret` Modal Secrets, detached runs up to 24 h.

### Changed

- Train in bf16 mixed precision with TF32 (`--bf16 --tf32 true`) in all run
  scripts. Runs were full fp32 before, which leaves most of an A100 unused;
  weights and optimizer state stay fp32. Needs an Ampere or newer GPU.
- The run scripts forward extra arguments (`"$@"`) to training; a repeated
  option overrides the script's value (e.g. `--output_dir_root`).
- Moved the run scripts (`train.sh`, `debug.sh`, `train_mixed_nllb_200k.sh`,
  `train_backtranslated.sh`) to `scripts/`: run `./scripts/train.sh`. They
  work from the repository root as before.

- New `--model_dtype` option (e.g. `bfloat16`) to load the model weights in a
  lower precision. Off by default. Meant for smoke tests on small GPUs: on a
  4 GB RTX 3050 the pipeline runs with `--model_dtype bfloat16 --optim sgd
  --gradient_checkpointing --max_length 64`, with the CTranslate2 conversion
  and FLORES+ eval run separately (training plus conversion exceed 7.5 GB RAM).
- Train on `madoss/moore-web-parallel` (config `mos-fra`) by default. New
  `--dataset_config` and `--dataset_revision` options; `train.sh` and
  `debug.sh` pin `v1.0.0`.
- Tokenize whole batches with the tokenizer's own `src_lang`/`tgt_lang`.
  The per-row `src_lang`/`tgt_lang` passed to the tokenizer call were
  silently ignored by NLLB (tokens were still `fra_Latn`/`mos_Latn`, so past
  runs were unaffected), and the per-row loop was slow.
- The in-training validation subset (`--validation_size`) is now a seeded
  shuffle of the validation split instead of its first rows, which can be
  grouped by source (in `moore-web-parallel` v1.0.0 the first 500 rows are
  64% `conseils` and contain no MAFAND).

### Fixed

- `main()` pushed the model to the Hub even without `--push_to_hub`, so
  `debug.sh` uploaded to `madoss/nllb-dry-run`. The push now follows the
  flag; `train.sh` and the other full-run scripts pass it.
- HF inference now sets `tokenizer.src_lang` before tokenizing, as the
  CTranslate2 path already did. Passing `src_lang` to the tokenizer call was
  ignored, so a non-default `--src_lang` still encoded the input as
  `fra_Latn`. The default French → Mooré direction was unaffected.

- Replaced Trackio experiment reporting with Weights & Biases to avoid the
  Trackio/Hugging Face Hub push failure tracked in
  <https://github.com/gradio-app/trackio/issues/544>.

## [0.1.0] - 2026-05-09

Initial release of `mt-training`, a toolkit for fine-tuning and evaluating
NLLB-200 for French to Moore machine translation.

### Added

- Training pipeline for `facebook/nllb-200-distilled-600M` on the
  `madoss/fr-mos-final-data` dataset.
- HuggingFace Hub publishing support with configurable model, dataset, language,
  output, and repository settings.
- BLEU and chrF++ metrics during training, with chrF++ used to select the best
  checkpoint.
- Evaluation CLI for HuggingFace, local, S3/Tigris, and FLORES+ datasets.
- Batch inference and interactive translation CLI for French to Moore.
- Optional CTranslate2 conversion and inference support.
- Optional quickmt-based backtranslation workflow for French data augmentation.
- `uv` project setup, shell entry points, training scripts, and example configs.
