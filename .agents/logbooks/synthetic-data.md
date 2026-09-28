# Synthetic and backtranslated data (`backtranslate.py`, `merge_datasets.py`, `scripts/train_mixed_nllb_200k.sh`, `scripts/train_backtranslated.sh`)

Experiments adding non-authentic French–Mooré pairs to the ~34k authentic
ones: NLLB's en → mos training data with the English side translated to
French (HY-MT), and backtranslation of French monolingual data (quickmt).
Run scores are in [experiments.md](experiments.md).

## 2026-09-28

- **These scripts still train on the old datasets**
  (`fr-mos-final-data-nllb-top200k-dedup`,
  `fr-mos-final-data-backtranslated-merged`), built from
  `fr-mos-final-data`. Rebuilding them on `moore-web-parallel` v1.0.0 is
  needed before comparing with new runs.

## 2026-04-21 – 2026-06-02

- **Adding 200k pivoted pairs made the model worse**, even deduplicated.
  Main hypothesis: NLLB already saw those Mooré sentences in pre-training.
  Planned next: lower synthetic ratios, then a from-scratch transformer.
  Scores, hypotheses and the planned runs are in
  [experiments.md](experiments.md).
