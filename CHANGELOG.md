# Changelog

All notable changes to this project will be documented in this file.

## [Unreleased]

### Changed

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
