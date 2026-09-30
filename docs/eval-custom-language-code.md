# Evaluating a model that uses its own language code

Some fine-tuned NLLB models add their own target language token instead of
using NLLB's `mos_Latn`. For example, `burkimbia/BIA-NLLB-600M-10E-CD-CRCL`
adds `moor_Latn` (see its `added_tokens.json`).

`mt_training.eval` can't score these as is:

- it always loads the base NLLB tokenizer, which doesn't have the new token;
- `--tgt_lang` picks both the code forced at decoding and the benchmark files,
  and FLORES+ and Bouquet only have `mos_Latn` files.

Until `eval.py` has options for this, wrap its two functions and run it with the
usual arguments. Keep `--tgt_lang` at `mos_Latn` so the benchmark files are
found. The model then decodes with `MODEL_TGT_LANG`:

```sh
uv run python - --model <CT2 dir or HF model> --dataset facebook/bouquet \
    --normalize_references true --show_samples 0 \
    --output evaluations/<name>-bq-sent.csv <<'EOF'
from mt_training import eval as ev
from mt_training.inference import load_model, translate_batch

TOKENIZER = "burkimbia/BIA-NLLB-600M-10E-CD-CRCL"  # repo with the added token
MODEL_TGT_LANG = "moor_Latn"

ev.load_model = lambda model, dtype=None: load_model(model, tokenizer_name=TOKENIZER, dtype=dtype)
ev.translate_batch = lambda texts, model, tok, src, _tgt, *rest: translate_batch(
    texts, model, tok, src, MODEL_TGT_LANG, *rest
)
ev.main()
EOF
```

Any `eval.py` option works after `-`, e.g. `--bouquet_level paragraph_level
--max_new_tokens 384` for Bouquet paragraphs, or the default FLORES+ dataset.
Always pass `--output`: without it nothing is saved.

For a CT2 model, convert the Hub checkpoint first:

```sh
uv run python -m mt_training.convert_ct2 --model <hub model> --output_dir <CT2 dir>
```

The converted vocabulary includes the added token, so only the tokenizer needs
to change.

Also score the model with plain `mos_Latn` and compare the two with
`mt-training compare`. An added token doesn't prove the model uses it: for
BIA-CRCL, `mos_Latn` scored higher (see `.agents/logbooks/experiments.md`,
2026-09-30).
