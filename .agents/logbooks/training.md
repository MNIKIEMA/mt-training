# Training (`train.py`, `train.sh`, `debug.sh`)

Fine-tunes `facebook/nllb-200-distilled-600M` for French → Mooré with
`Seq2SeqTrainer`: loads a Hub dataset, tokenizes, evaluates BLEU/chrF++ on a
validation subset during training (chrF++ picks the best checkpoint), then
runs the in-domain `test` split, CTranslate2 conversion and FLORES+ (see
[evaluation-inference.md](evaluation-inference.md)).

## 2026-09-28 (moore-web-parallel v1.0.0)

- **Default dataset is now `madoss/moore-web-parallel`**, config `mos-fra`,
  pinned to `v1.0.0` in `train.sh` and `debug.sh` (`--dataset_config`,
  `--dataset_revision`). 39,038 / 2,491 / 2,573 rows; the extra columns
  (`id`, `original_lang`, `doc_id`, `reviewed`, scores) pass through and are
  dropped at tokenization. The mixed and backtranslated scripts still train on
  the old `fr-mos-final-data-*` datasets. Results before and after this
  switch are not comparable: different data and a different validation subset.
- **`src_lang`/`tgt_lang` passed to the tokenizer call are silently
  ignored** by the NLLB tokenizer (transformers 5.5.0): passing
  `eng_Latn`/`dyu_Latn` still produced `fra_Latn`/`mos_Latn`. Language tokens
  come only from the tokenizer's own `src_lang`/`tgt_lang`, set in `main()`,
  so past runs were tokenized correctly. The per-row language columns and
  per-row loop are gone; batched tokenization gives identical ids (checked on
  2,000 rows). To change direction, change the tokenizer, not the call.
- **The validation subset was the first N rows.** The v1.0.0 validation split
  is stored grouped by source: its first 500 rows were 318 `conseils`, 0
  MAFAND, so best-checkpoint selection would have tracked government-report
  language only. Now a shuffle with `--seed` before taking
  `--validation_size` rows (500 → 315 MAFAND, 125 conseils, rest mixed).
- **Nothing is truncated at `max_length` 256** on v1.0.0: longest input 199
  tokens, longest label 210 (2,000-row sample). Mooré runs ~0.40 tokens/char
  with the NLLB tokenizer, French ~0.31.
- **`trainer.push_to_hub()` ran unconditionally**, so `debug.sh` would have
  uploaded to `madoss/nllb-dry-run`. It now follows `--push_to_hub`.
- **Smoke test on a 4 GB RTX 3050 laptop GPU (7.5 GB RAM) works with**
  `--model_dtype bfloat16 --optim sgd --gradient_checkpointing --max_length 64
  --per_device_train_batch_size 4 --report_to none`. 30 steps take ~30 s.
  What doesn't fit: default fp32 + AdamW needs ~10 GB; bf16 + Adafactor OOMs
  at the first optimizer step (Adafactor upcasts each parameter to fp32, and
  the shared 256k-token embedding alone is 1 GB in fp32). Training and CT2
  conversion in one process exceed 7.5 GB RAM (OOM-killed, exit 137): run the
  conversion and FLORES+ separately on the saved model. bf16 weights + SGD
  barely learn: pipeline checks only, never real runs.
- **"missing keys `model.encoder.embed_tokens`"** is logged when the best
  checkpoint is reloaded. Likely the tied encoder/decoder/shared embedding;
  conversion and translation afterwards were fine. Not investigated further.

## 2026-05 – 2026-06 (from earlier notes)

- **`--train_sampling_strategy group_by_length`** (with dynamic padding)
  brought a full run from ~22 h to ~5 h.
- **Trackio replaced by W&B** (2026-05-10): `trackio.sync(sdk="static",
  private=True)` wrote the HF token into the Space's `config.json`, and the Hub
  rejected the push (details in `trackio_bug.md`, `trackio-issue.md`). Keep
  `wandb.finish()` after post-training eval and Hub push.
- **`NOTE.md`: "transformers==5.5.3 does not work for NLLB"** -- recorded
  without detail. 5.5.0 (current lock) works. Check before upgrading.
