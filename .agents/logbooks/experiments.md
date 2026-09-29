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

## 2026-09-29 (v2: NLLB <unk> fix, both directions)

- **`nllb-600m-FrMos-mwp-v2` and `nllb-600m-MosFr-mwp-v2`**: same scripts,
  data (`moore-web-parallel` v1.0.0) and hyperparameters as v1, plus
  `normalize_for_nllb` before tokenization (commit `3303a25`). Trained in
  parallel on Modal.
- **Their own post-training scores are not comparable with v1**: the test
  metric is computed against tokenized labels, which for v1 had ’ and « »
  deleted (<unk>) and for v2 contain ' and "; FLORES+/Bouquet references keep
  ’ « », which penalizes v2 for writing ' and ". So v1 and v2 were re-scored
  with the same references, normalized on both sides
  (`eval.py --normalize_references true`, new): CT2 int8 on the local RTX
  3050, 128 new tokens (sentences), 384 (Bouquet paragraphs), 256 (test);
  in-domain test = `moore-web-parallel` test split. chrF++, v2 − v1 with 95%
  paired bootstrap CI (`mt-training compare`):

  | | FLORES+ (1,012) | Bouquet sent. (854) | Bouquet para. (198) | test (2,573) |
  | --- | --- | --- | --- | --- |
  | mos → fra v1 → v2 | 30.01 → 30.46, +0.45 [+0.12, +0.78] | 33.12 → 34.12, +1.00 [+0.54, +1.47] | 38.72 → 38.86, +0.14 [−0.43, +0.74] | 38.56 → 38.53, −0.03 [−0.30, +0.21] |
  | fra → mos v1 → v2 | 22.85 → 22.68, −0.18 [−0.42, +0.07] | 29.05 → 28.74, −0.31 [−0.73, +0.10] | 34.03 → 34.01, −0.02 [−0.46, +0.42] | 32.77 → 32.69, −0.08 [−0.34, +0.16] |

- **Mooré → French: the fix works.** Outputs with glued elisions ("cest",
  "quil"): FLORES+ 198 -> 3, Bouquet sentences 96 -> 5, paragraphs 22 -> 4,
  test 394 -> 9; outputs with an apostrophe roughly double (FLORES+ 361 ->
  869). Significant gains on FLORES+ and Bouquet sentences.
- **French → Mooré: no significant change** (all CIs include 0). Mooré rarely
  uses ’ (280 training rows), and straight " was already common in the
  training text, so v1 wrote quotes as often as the references (Bouquet
  sentences: 44 v1 / 43 v2 outputs with quotes, 37 references).
- **Use v2 in both directions** (MosFr-v2 for backtranslation).
- **Normalized references raise absolute scores a little** (v1 fra→mos
  FLORES+ 22.85 vs 22.56 with raw references): compare runs scored the same
  way only.

## 2026-09-29 (Mooré → French model, for backtranslation)

- **`madoss/nllb-600m-MosFr-mwp-v1`** (private), W&B `4a7ha86u`,
  `scripts/train_mos_fra.sh` (= `train.sh` hyperparameters, columns and NLLB
  codes swapped), `moore-web-parallel` v1.0.0, bf16, Modal A100-80GB, 8
  epochs, 29 min end to end, code of commit `4de55b4`. In-domain test (2,573):
  20.55 BLEU / 38.48 chrF++.
- **Clearly better than the untuned NLLB into French**, same pipeline (CT2
  int8; FLORES+ on Modal via `--flores-model`, Bouquet on the local RTX 3050
  with `eval.py --src_lang mos_Latn --tgt_lang fra_Latn`, 128 new tokens for
  sentences, 384 for paragraphs; CIs from `mt-training compare`):

  | Mooré → French | untuned NLLB | `nllb-600m-MosFr-mwp-v1` | chrF++ gain [95% CI] |
  | --- | --- | --- | --- |
  | FLORES+ devtest (1,012), BLEU / chrF++ | 7.51 / 26.74 | 7.83 / 29.93 | +3.19 |
  | Bouquet sentences (854), BLEU / chrF++ | 7.03 / 24.80 | 12.43 / 32.42 | +7.62 [+6.84, +8.43] |
  | Bouquet paragraphs (198), BLEU / chrF++ | 7.24 / 26.37 | 14.04 / 37.30 | +10.93 [+9.46, +12.25] |

  Output/reference length on Bouquet paragraphs: untuned 0.72, fine-tuned
  0.97: the untuned model drops content on long inputs in this direction too.
  Larger gains than French → Mooré (FLORES+ +1.78, Bouquet sentences +2.62):
  NLLB's French decoder is strong, so learning to read Mooré pays off more.
- **Use it for backtranslating `moore-web-mono` v1.1.0** (8,817 sentences).

## 2026-09-29 (burkimbia/mt-benchmark-public submission)

- **Translated the blind benchmark with `nllb-600m-FrMos-mwp-v1`** (CT2
  int8, local RTX 3050, beam 4, no-repeat 3, `max_new_tokens` 384 since
  sources reach 845 chars): 1,475 French sources → Mooré, ids 0–1474, no
  empty output. Submission CSV (`id,translation`):
  `submissions/nllb-600m-FrMos-mwp-v1.csv` (untracked). Domains:
  arts_technology 300, daily_life 300, religious 297, administrative 291,
  health 233, unknown 54; a `weight` column (1–3) is given, meaning not
  documented. No references, no submission instructions in the card.
- **The benchmark shares documents with our training data.** `mt-training
  overlap` (normalized exact match on the French side, lower bound): 214 of
  1,475 rows. By length: 47 of 1–2 words and 20 of 3–5 (dictionary
  headwords, fragments: not a real leak); **147 of 6+ words**: 92
  `conseils` (administrative, 67 of them 11+ words) and 54 `kade`/`sida`
  (health). About a third of `administrative` and a fifth of `health` match
  exactly; more of those domains likely come from the same documents.
  `religious`, `arts_technology`, `daily_life` are essentially clean (3 short
  matches). Scores on administrative and health will overstate the model;
  report per domain, or train a decontaminated model (drop the matching
  sources, or whole `conseils`/`kade`/`sida` documents) for a fair entry.
  Match list: `submissions/nllb-600m-FrMos-mwp-v1.overlap.csv`.
- **Leaderboard (2026-09-29), same translation settings for both:**

  | Model | chrF | BLEU |
  | --- | --- | --- |
  | untuned `facebook/nllb-200-distilled-600M` | 35.10 | 5.22 |
  | `nllb-600m-FrMos-mwp-v1` | 42.68 | 7.92 |

  Gain +7.58 chrF, +2.70 BLEU: much more than on the clean benchmarks
  (chrF++: FLORES+ +1.78, Bouquet sentences +2.62, paragraphs +8.42). Three
  effects mixed: the 147 shared sentences (same documents), domain match
  (administrative and health text is most of our training data), and
  complete translations of long inputs (the untuned outputs are 4% shorter
  here). The transferable gain is likely nearer the FLORES+/Bouquet numbers;
  per-domain leaderboard scores or a decontaminated retrain would measure it.
  Leaderboard chrF is plain chrF (not chrF++), possibly weighted by the
  undocumented `weight` column. Submission files:
  `submissions/nllb-200-distilled-600M-untuned.csv`,
  `submissions/nllb-600m-FrMos-mwp-v1.csv` (untracked).

## 2026-09-29 (Bouquet: untuned vs June vs v1.0.0 model)

- **Bouquet `fra_Latn-mos_Latn` test confirms the ranking, more clearly than
  FLORES+.** CT2 int8 on the local RTX 3050, `eval.py --dataset
  facebook/bouquet` (beam 4, no-repeat 3; `--max_new_tokens` 128 for
  sentences, 384 for paragraphs, same for every model: 55 of 198 paragraph
  references exceed 128 Mooré tokens). CIs: `mt-training compare`, 1,000
  resamples, seed 0.

  | Bouquet test | untuned NLLB | `nllb-600m-FrMos` (June) | `nllb-600m-FrMos-mwp-v1` |
  | --- | --- | --- | --- |
  | sentences (854), chrF++ / BLEU | 26.09 / 9.18 | 27.12 / 8.33 | 28.71 / 10.56 |
  | paragraphs (198), chrF++ / BLEU | 25.34 / 7.49 | 23.63 / 5.92 | 33.77 / 12.46 |
  | paragraphs, output/reference length | 0.71 | 0.59 | 0.94 |

  chrF++ gains [95% CI]: sentences new − untuned +2.62 [+1.91, +3.30], new −
  June +1.60 [+1.14, +2.08] (positive in every length quartile), June −
  untuned +1.02 [+0.32, +1.74]; paragraphs new − untuned +8.42 [+7.17,
  +9.73], new − June +10.14 [+8.60, +11.69], June − untuned −1.71 [−3.10,
  −0.39].
- **The paragraph gap is completeness.** The untuned and June models stop
  early on paragraphs (0.71 and 0.59 of the reference length); the new model
  translates the whole paragraph (0.94). Likely from v1.0.0's long
  multi-sentence pairs (`expert`, `news`, reviewed units), not verified.
  Same pattern as the long FLORES+ quartile.
- **June model re-run on FLORES+ via `modal_train.py::flores`**: 2.72 /
  21.01 (post-training run: 2.76 / 20.96), so the FLORES+ comparison stands.
  Its CT2 model is on the volume at `eval/madoss--nllb-600m-FrMos-ct2-int8`.
- **Bouquet is the better external check for this data**: conversational
  and how-to text, 8 domains, sentence and paragraph levels, CC BY 4.0, and
  checked free of overlap with the `expert` rows. Use the test split only for
  reporting; dev is available for choices.

## 2026-09-29 (sentence length: FLORES+ vs moore-web-parallel v1.0.0)

- **Training rows are much shorter than FLORES+, but long ones exist; most
  are `conseils`.** French characters and Mooré NLLB tokens (no special
  tokens):

  | Set | Pairs | fr chars median (p90) | mos tokens median (p90) |
  | --- | ---: | --- | --- |
  | FLORES+ devtest | 1,012 | 149 (223) | 39 (61) |
  | v1.0.0 train | 39,038 | 42 (228) | 11 (70) |
  | v1.0.0 validation | 2,491 | 134 (255) | 35 (73) |
  | v1.0.0 test | 2,573 | 123 (271) | 29 (73) |

  Per source (train), fr chars median / mos tokens median:
  `lexicon_entries` 9 / 2 (16,556 rows), `du-moore` 10 / 5, `digital-terms`
  12 / 7, `lexicon` 41 / 11, `mos-contes-volume-5` 76 / 23, `kade` 77 / 22,
  `abcburkina-contes` 79 / 21, `sida` 85 / 27, `udhr` 132 / 47, `mafand`
  137 / 34, `messages-nouvel-an` 150 / 37, `conseils` 163 / 50 (12,221 rows),
  `digital-defs` 176 / 44, `news` 189 / 44, `expert` 329 / 87.

- **9,590 train rows (24.6%) are at least as long as the FLORES+ median
  (149 French chars); 7,011 of them (73%) are `conseils`.** Other long rows:
  `mafand` 1,132, `news` 841, `expert` 296, `digital-defs` 80,
  `mos-contes-volume-5` 79, `kade` 65, `udhr` 28, `messages-nouvel-an` 24,
  `sida` 16. By tokens `conseils` weighs more still (long rows, 12,221 of
  them).
- **FLORES+ by length (untuned NLLB vs `nllb-600m-FrMos-mwp-v1`, CT2 int8,
  run locally on an RTX 3050 laptop GPU, same `eval.py` settings; quartiles
  of French length; chrF++ gain with 95% paired bootstrap CI, 1,000
  resamples, seed 0; `mt-training compare --baseline base.csv --candidate
  new.csv` on the two `eval.py --output` files):**

  | Quartile (fr chars) | n | chrF++ untuned → new | Gain [95% CI] | out/ref length untuned → new |
  | --- | ---: | --- | --- | --- |
  | Q1 (47–117) | 253 | 20.95 → 22.18 | +1.23 [+0.43, +2.04] | 0.98 → 0.99 |
  | Q2 (117–149) | 253 | 20.64 → 21.95 | +1.32 [+0.60, +2.06] | 0.93 → 0.96 |
  | Q3 (149–188) | 253 | 20.72 → 22.48 | +1.75 [+0.98, +2.57] | 0.89 → 0.97 |
  | Q4 (188–415) | 253 | 20.81 → 23.16 | +2.35 [+1.59, +3.05] | 0.85 → 0.98 |
  | All | 1,012 | 20.77 → 22.55 | +1.78 [+1.40, +2.16] | 0.90 → 0.97 |

  Local totals differ slightly from Modal (untuned 2.94 / 20.77 here vs
  3.08 / 20.97; new 3.38 / 22.55 vs 3.33 / 22.56): int8 on another GPU.
  Compare within one device.
- **The gain is significant and grows with length.** The longest quartile
  gains most (+2.35), largely by translating the whole sentence: the untuned
  model's outputs are 15% shorter than the references there, the fine-tuned
  model's are the right length. Not the 128-token limit (FLORES+ Mooré p90 is
  61 tokens). Per-quartile BLEU is too noisy to read (Q1 3.10 → 2.69).
- **Reading (corrected):** length is not the bottleneck; every quartile
  stays around 22–23 chrF++, so what limits FLORES+ is quality across the
  board, most likely domain and vocabulary (long training sentences are
  mostly government-report language) and possibly orthography. The
  candidates stand: backtranslation of real Mooré text (raamde Mooré side,
  mooreburkina, HPLT) or down-weighting `conseils`, not round-trip synthetic
  data from NLLB (see H2).

## 2026-09-28 (moore-web-parallel v1.0.0, bf16, Modal)

- **First run on the new dataset: in-domain gain, and better than the June
  model on FLORES+.**
  `madoss/nllb-600m-FrMos-mwp-v1`, W&B run `ggomavbg`
  (`moore-web-parallel-v1.0.0-bf16`). `scripts/train.sh` on
  `madoss/moore-web-parallel` `mos-fra` v1.0.0 (39,038 train rows), bf16 +
  TF32, Modal A100-80GB, 8 epochs = 2,440 steps. Cancelled once at step ~626
  and resumed from `checkpoint-610` (see [training.md](training.md)). Wall
  time of the resumed part: 58 min, including test eval, CT2 conversion,
  FLORES+ and Hub push. Metric: **chrF++**.

  In-training validation (500 shuffled rows, seed 42: ~315 MAFAND,
  ~125 conseils, rest mixed):

  | Epoch | Train loss | Val loss | BLEU | chrF++ |
  | --- | --- | --- | --- | --- |
  | 1 | 5.22 | 2.911 | 6.04 | 27.43 |
  | 2 | 4.35 | 2.660 | 7.58 | 29.95 |
  | 3 | 3.82 | 2.565 | 8.54 | 31.30 |
  | 4 | 3.21 | 2.493 | 9.60 | 32.05 |
  | 5 | 2.76 | 2.453 | 9.61 | 31.97 |
  | 6 | 2.71 | 2.432 | 9.73 | 32.60 |
  | 7 | 2.74 | 2.429 | 10.09 | 32.86 |
  | 8 | 2.26 | 2.428 | 10.35 | 32.90 |

  Final (best = epoch 8):

  | Eval | Examples | Model | BLEU | chrF++ |
  | --- | --- | --- | --- | --- |
  | In-domain `test` (v1.0.0) | 2,573 | HF, trainer generation | 9.88 | 32.19 |
  | FLORES+ devtest | 1,012 | CT2 int8 | 3.33 | 22.56 |

- **Better than the previous model on FLORES+: +0.57 BLEU, +1.60 chrF++.**
  Reference: `madoss/nllb-600m-FrMos`, trained 2026-06-02 by the same
  post-training pipeline (CT2 int8, same `eval.py` CT2 path, chrF++) on
  `fr-mos-final-data`, fp32. Its FLORES+ was 2.76 BLEU / 20.96 chrF++ (below
  the untuned NLLB's 2.94 BLEU from 2026-04-17); this run 3.33 / 22.56. Data
  and precision both changed, so the gain can't be split between them; bf16
  normally doesn't move quality. No significance test yet.
- **Untuned NLLB through the same pipeline** (`modal_train.py::flores`,
  CT2 int8 on Modal A100-80GB, same `eval.py` settings): 3.08 BLEU / 20.97
  chrF++. So on FLORES+ the June fine-tune added nothing (20.96 chrF++, BLEU
  2.76 below the untuned model) and this run is the first to beat the untuned
  model: +1.59 chrF++, +0.25 BLEU. Predictions of the untuned model:
  `mt-training-outputs:eval/facebook--nllb-200-distilled-600M-flores_plus-devtest.csv`.

  | FLORES+ devtest (1,012), CT2 int8 | BLEU | chrF++ |
  | --- | --- | --- |
  | untuned `facebook/nllb-200-distilled-600M` | 3.08 | 20.97 |
  | `madoss/nllb-600m-FrMos` (2026-06-02) | 2.76 | 20.96 |
  | `madoss/nllb-600m-FrMos-mwp-v1` (this run) | 3.33 | 22.56 |
- **In-domain test scores are not comparable across the two runs**: the
  June model's 27.78 chrF++ is on `fr-mos-final-data`'s test split, this
  run's 32.19 on v1.0.0's, and the June training data may overlap v1.0.0
  test rows.
- **Still improving at epoch 8** (val loss and chrF++), early stopping never
  triggered; gains were small over the last three epochs. A longer run is
  cheap to try (~25 min of training per 8 epochs).
- **Next:** score v1.0.0 `test` per source for this model and the untuned
  NLLB, to see where fine-tuning helps; paired bootstrap on FLORES+ against
  the June model.

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
  HY-MT. Script `scripts/train_mixed_nllb_200k.sh`.

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
