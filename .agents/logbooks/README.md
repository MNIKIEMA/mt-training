# Logbooks

These track how training and evaluation evolved: decisions made, what real
runs revealed, fixes applied, and limitations left open. The goal is that the
next session (human or agent) doesn't have to re-derive a quirk we already
found. `git log` has the diffs; `CHANGELOG.md` has the user-facing changes;
logbooks have the *why* and the numbers.

## Convention: one logbook per module

A logbook maps 1:1 to a module or workflow, not to a single run. Scores of
full runs all go in `experiments.md`, so runs can be compared in one place.

- [`training.md`](training.md) -- `train.py`, `train.sh`, `debug.sh` (NLLB fine-tuning with `Seq2SeqTrainer`, dataset loading, tokenization, in-training eval, Hub push).
- [`evaluation-inference.md`](evaluation-inference.md) -- `eval.py`, `inference.py`, `convert_ct2.py` (BLEU/chrF++, FLORES+, HF and CTranslate2 translation).
- [`experiments.md`](experiments.md) -- all training scripts (scores of full runs, hypotheses, planned experiments; converted from `exp_res.md`).
- [`synthetic-data.md`](synthetic-data.md) -- `backtranslate.py`, `merge_datasets.py`, `train_mixed_nllb_200k.sh`, `train_backtranslated.sh` (synthetic and backtranslated data experiments).

Add a new logbook when a new module or workflow is added, not for each run.

## Entry format

Dated, terse, newest entry on top. Prefer "what we learned / decided" over
"what the diff was". Give scores with their dataset, split, number of
examples, and model format (HF or CT2), or they can't be compared later.

```markdown
## 2026-09-28

- Finding or decision, one or two sentences.
- Why it matters / what to check next.
```
