# Synthetic and round-trip data (`round_trip.py`, `merge_datasets.py`, `scripts/train_mixed_nllb_200k.sh`, `scripts/train_round_trip.sh`)

Experiments adding non-authentic French–Mooré pairs to the ~34k authentic
ones: NLLB's en → mos training data with the English side translated to
French (HY-MT), and round-trip translation of French through English
(quickmt, `round_trip.py`; called "backtranslation" until 2026-09-29).
Run scores are in [experiments.md](experiments.md).


## 2026-09-29 (backtranslation with MosFr-v2)

- **Output**: `data/bt/moore-web-mono-v1.1.0-bt-MosFr-v2.jsonl` (not in git),
  all 8,817 sentences of moore-web-mono v1.1.0 translated by
  `madoss/nllb-600m-MosFr-mwp-v2` (CT2 int8, beam 4, RTX 3050, 463 s).
  Kept 8,764 (99.4%); dropped 49 `length_ratio`, 2 `loop`, 2 `moore_letters`.
  `bt_model` records the Hub id (the run used a local CT2 conversion of it).
- **Supersedes `moore-web-mono-bt.jsonl`** (MosFr-v1), which lost apostrophes:
  same ids, sentences with an apostrophe 2,775 → 6,228, glued elisions
  ("Cest", "quil", "dun"…) 885 → 3, sentences with quotes 293 → 1,025.
- **HF inference in fp32 ran out of memory** on the 3050 (fine-tuned
  checkpoints are fp32; beam 4 × batch 32). Added `--dtype bfloat16`: fits,
  about 6 sentences/s versus 19 with CT2 int8, so CT2 stays the local choice.
- **Scores** (moore-web `annotate`, default batch sizes; not used to filter):
  `data/bt/moore-web-mono-v1.1.0-bt-MosFr-v2-scored.jsonl` adds
  `laser_score` (LASER2 fra / LASER3 mos) and `comet_qe`
  (`McGill-NLP/ssa-comet-qe`). Kept pairs: LASER p5/median/p95
  0.69 / 0.83 / 0.90; COMET-QE 0.51 / 0.63 / 0.72.
  - The two barely agree: Spearman 0.24; of the bottom 10% by each, 153 of
    876 pairs are shared (88 expected by chance).
  - LASER's lowest (down to −0.14) are correct translations: Galatians 3:8
    and 3:5, a purification ceremony. Their COMET-QE is 0.55–0.67. LASER3's
    Mooré encoder is the weak side, so LASER alone is a poor filter here.
  - COMET-QE's lowest (≈0.35–0.39) are real problems: mistranslations
    ("pêche laitière … poussins"), garbled Mooré sources
    ("Rɩk-y n dɩk-y-yã-yã-a…"), untranslated names ("Clash cymbals").
  - The rule-based drops have the same median COMET-QE (0.62) as kept
    pairs: the rules and COMET-QE catch different failures.
- **Running the scorers on the RTX 3050 laptop (7.5 GB RAM)**: batched LASER3
  fails in fairseq's transformer fast path ("Mask Type should be defined");
  loading ssa-comet-qe (XLM-R large) was killed by the kernel OOM killer
  (~4.1 GB RSS) while VS Code was open. The GPU was not the limit. It ran
  with VS Code closed, batch size 4.
- Next: French → Mooré on moore-web-parallel v1.1.0 + all kept pairs,
  compared with FrMos-v2; then an ablation filtering on COMET-QE rather
  than LASER.

## 2026-09-29 (backtranslation tool)

- **`mt_training/backtranslate.py`** (`mt-training backtranslate`,
  `scripts/backtranslate.sh`, runs on Modal through the launcher, which adds
  `--output_dir_root /outputs/`). Pairs keep the Mooré sentence's content id,
  `source: bt-<source>`, `original_lang: mos`, `reviewed: false`,
  `bt_model`, and a `drop_reason`; nothing is dropped at write time, so the
  checks can change without re-translating. Output is appended per batch and
  existing ids are skipped (resume). French is checked without a language-ID
  model: Mooré-only letters (ɛ ɩ ʋ ã ẽ ĩ õ ũ) in the output mean untranslated
  Mooré.
- **Tried with the untuned NLLB (CT2 int8, RTX 3050) on the first 500
  moore-web-mono sentences**: 1 s per 30, 494 kept, 6 `length_ratio` drops,
  all real omissions (French far shorter than the Mooré). Kept French/Mooré
  length ratio: p5 0.65, median 0.96, p95 1.48, so the [0.5, 3.0] bounds only
  catch outliers. Fluent French but factual slips (two birth dates for the same
  person, "législatives" for a presidential election).
- **moore-web-mono contains NLLB output**: 25 sentences on 10 incubator pages
  carry leaked NLLB language tags ("… mos_Latnmos_Latnmos_Latn be be be").
  Those articles were machine-translated with NLLB; training NLLB on its own
  Mooré is H2 again. Handled in moore-web: tagged pages dropped whole;
  `moore-web-mono` v1.1.0 (8,817 sentences) is the backtranslation input and
  the tool's default revision.

## 2026-09-29 (rename)

- **`backtranslate.py` → `round_trip.py`, `train_backtranslated.sh` →
  `train_round_trip.sh`.** The workflow round-trips French through English
  to paraphrase the source side; it teaches nothing new about Mooré, like
  the pivoted NLLB data (H2 in experiments.md). "Backtranslation" is now
  reserved for Mooré monolingual text → French (`madoss/moore-web-mono`
  with a Mooré → French model). Hub names (`fr-mos-final-data-backtranslated…`,
  `fr-mos-backtranslated-merged-mt`) keep the old word.
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
