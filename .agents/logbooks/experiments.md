# Experiments (all training scripts; scores of full runs)

One place for the scores of full training runs and what they taught us.
Module-level findings (bugs, memory, tokenization) go in
[training.md](training.md), [evaluation-inference.md](evaluation-inference.md)
and [synthetic-data.md](synthetic-data.md).

Unless an entry says otherwise, scores are CTranslate2 models, fr → mos,
`beam_size=4`, `no_repeat_ngram_size=3`, on:

- **FLORES+**: `openlanguagedata/flores_plus` `devtest`, 1,012 sentences;
- **S3 ref**: `s3://burkimbia-store/evaluation/references/MT/dataset`,
  1,471 sentences, a private in-domain reference set, dropped from
  post-training eval on 2026-05-25.

**Metric caveat:** the entries up to 2026-06-02 were recorded as **chrF**.
The metric became chrF++ on 2026-05-06 (`91b4bfa`), so treat them as plain
chrF: not comparable with chrF++ scores from later runs. BLEU and TER are
comparable.

Converted from `exp_res.md` on 2026-09-28; dates come from that file and the
timestamps of the prediction files next to it.

## 2026-04-21 – 2026-06-02 (deduplicated mixed data, stage 2)

- **Deduplicating the mixed data helped a little, still below
  authentic-only.** Dataset `madoss/fr-mos-final-data-nllb-top200k-dedup`.
  A second stage (`nllb-top200k-ct2-stage2`) changed almost nothing; it
  early-stopped at 4,000 steps.

  | Model | FLORES+ BLEU / chrF / TER | S3 ref BLEU / chrF / TER |
  | --- | --- | --- |
  | `nllb-top200k-ct2` (dedup) | 2.85 / 22.39 / 96.28 | 7.96 / 27.51 / 87.24 |
  | `nllb-top200k-ct2-stage2` | 2.82 / 22.61 / 96.33 | 7.96 / 27.76 / 87.47 |

## 2026-04-21 (33k authentic + 200k synthetic)

- **More data made the model worse**, on both sets and all metrics.
  Training data: the 33,835 rows of `fr-mos-final-data` plus 200k rows from
  NLLB's en → mos training data, English side translated to French with
  HY-MT. Script `train_mixed_nllb_200k.sh`.

  | Model | FLORES+ BLEU / chrF / TER | S3 ref BLEU / chrF / TER |
  | --- | --- | --- |
  | `mixed-nllb-top200k-mt/checkpoint-9135-ct2` | 2.79 / 21.86 / 96.33 | 7.54 / 26.23 / 89.07 |

  Run with `mt-training eval --model /workspace/mixed-nllb-top200k-mt/checkpoint-9135-ct2
  --max_new_tokens 256` (add `--dataset openlanguagedata/flores_plus` for
  FLORES+). Predictions: `evaluations-mixed200k.csv`.

- **Why, three hypotheses:**
  - **H1, noisy French pivot (weakened).** The French side comes from HY-MT, a
    strong model, so it is probably good; unlikely to be the main cause.
  - **H2, data NLLB has already seen (primary).** NLLB was pre-trained on the
    original en → mos data. The same Mooré sentences with a French source
    teach it nothing new about Mooré; only the 33k authentic pairs do.
  - **H3, dilution.** Even with a good pivot, 200k synthetic against 33k
    authentic pairs pulls the model toward the synthetic distribution.

- **Next experiment A, planned, not run: NLLB with less synthetic data.**
  If scores improve as the synthetic share drops, H3 matters and NLLB
  fine-tuning stays the path; if they stay flat, H2 dominates: go to B.

  | Authentic | Synthetic | Total | Ratio |
  | --- | --- | --- | --- |
  | 33k × 1 | 33k | ~66k | 1:1 |
  | 33k × 1 | 66k | ~99k | 1:2 |
  | 33k × 3 (oversampled) | 33k | ~132k | 3:1 |
  | 33k × 5 (oversampled) | 33k | ~198k | 5:1 |

- **Next experiment B, planned: a from-scratch transformer.** ~913k fr → mos
  pairs (880k pivoted from NLLB's en → mos data + 33k authentic, oversampled
  as a quality anchor). H2 doesn't apply to a model that never saw the data,
  880k pairs is enough for a small model (6-layer encoder-decoder, ~60M
  parameters, Opus-MT scale), and a dedicated fr → mos model avoids
  interference from NLLB's 200 languages. Remaining risk: the pivot may bias
  the French source distribution. The untracked `eole.yaml` and `train.yaml`
  (6 layers, 8 heads, 512 hidden), created the same day, look like the start
  of this experiment.

## 2026-04-17 (baseline and first fine-tune)

- **Fine-tuning on 33k authentic pairs gains little on FLORES+.** Training
  data `madoss/fr-mos-final-data`, 33,835 rows. Predictions:
  `evaluations.csv`.

  | Model | FLORES+ BLEU / chrF / TER | S3 ref BLEU / chrF / TER |
  | --- | --- | --- |
  | `nllb-200-ct2` (baseline, not fine-tuned) | 2.94 / 22.4 / 99.9 | -- |
  | `/workspace/nllb-200-ct2` (fine-tuned) | 3.19 / 23.21 / 96.2 | 9.01 / 29.62 / 84.25 |
  | `nllb-200-finetuned-ct2`, `max_new_tokens` 256 | 3.13 / 23.23 / 96.44 | -- |

  `exp_res.md` doesn't say which model `/workspace/nllb-200-ct2` is; it is
  read as the fine-tuned model because the `max_new_tokens` note compares it
  with `nllb-200-finetuned-ct2`.

- **`max_new_tokens` 128 → 256 changed nothing** (BLEU 3.19 → 3.13, chrF
  23.21 → 23.23): output truncation was not the bottleneck.
