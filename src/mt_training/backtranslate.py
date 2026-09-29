"""
Backtranslation: Mooré monolingual sentences -> synthetic French.

Translates a Mooré dataset (default: `madoss/moore-web-mono` v1.1.0) into
French with a Mooré -> French model (HF id or CTranslate2 directory), and writes
French–Mooré pairs in the `moore-web-parallel` schema, marked as synthetic:
`original_lang = mos`, `reviewed = false`, `source = bt-<source>`, and the Mooré
sentence's content-based `id`, so French from different models can be compared
on the same sentences.

Every translated sentence is written, with `drop_reason` null when it passes
the checks below; filtering happens when the pairs are used, so the checks can
change without translating again:

- `empty`: no output;
- `copy`: the output repeats the Mooré input;
- `moore_letters`: 2+ Mooré-only letters (ɛ ɩ ʋ, ã ẽ ĩ õ ũ) in the output,
  i.e. Mooré left untranslated (French never uses them);
- `length_ratio`: French/Mooré character ratio outside [min, max];
- `loop`: an 8+ word output with fewer than `min_distinct_ratio` distinct
  words (generation loop).

Output is appended batch by batch, and ids already in the output file are
skipped, so an interrupted run resumes where it stopped.

Usage:
    python -m mt_training.backtranslate --model <mos-fra model or CT2 dir>
    python -m mt_training.backtranslate --model <model> --limit 20 --output bt-sample.jsonl
"""

import json
import re
import time
import unicodedata
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

import draccus

MOORE_ONLY_LETTERS = set("ɛɩʋãẽĩõũƐƖƲÃẼĨÕŨ")
_WORD = re.compile(r"[^\W\d_]+(?:[-'’][^\W\d_]+)*")
_PUNCT = re.compile(r"[^\w\s]")
_SPACES = re.compile(r"\s+")

TranslateFn = Callable[[list[str]], list[str]]


@dataclass
class BacktranslateConfig:
    model: str = field(default="", metadata={"help": "Mooré -> French model: Hub id or CT2 dir"})
    dtype: str | None = field(
        default=None,
        metadata={
            "help": "HF model weight dtype, e.g. bfloat16 (default: as saved; ignored for CT2)"
        },
    )
    dataset: str = field(default="madoss/moore-web-mono", metadata={"help": "Mooré dataset"})
    dataset_config: str | None = field(default="default", metadata={"help": "Dataset config"})
    dataset_revision: str | None = field(
        default="v1.1.0", metadata={"help": "Dataset tag or commit"}
    )
    split: str = field(default="train", metadata={"help": "Dataset split"})
    text_field: str = field(default="text", metadata={"help": "Mooré text column"})
    src_lang: str = field(default="mos_Latn", metadata={"help": "NLLB source code"})
    tgt_lang: str = field(default="fra_Latn", metadata={"help": "NLLB target code"})
    batch_size: int = field(default=32, metadata={"help": "Sentences per batch"})
    beam_size: int = field(default=4, metadata={"help": "Beam search width"})
    no_repeat_ngram_size: int = field(default=3, metadata={"help": "0 = disabled"})
    max_new_tokens: int = field(default=256, metadata={"help": "Max generated tokens"})
    limit: int | None = field(default=None, metadata={"help": "Translate only the first N rows"})
    output_dir_root: Path = field(default=Path("data/bt"), metadata={"help": "Output directory"})
    output: Path = field(
        default=Path("moore-web-mono-bt.jsonl"),
        metadata={"help": "Output JSONL (under output_dir_root)"},
    )
    min_length_ratio: float = field(default=0.5, metadata={"help": "Min French/Mooré char ratio"})
    max_length_ratio: float = field(default=3.0, metadata={"help": "Max French/Mooré char ratio"})
    min_distinct_ratio: float = field(
        default=0.55, metadata={"help": "Min distinct-word share (8+ words)"}
    )


def _normalize(text: str) -> str:
    text = unicodedata.normalize("NFKC", text).lower()
    return _SPACES.sub(" ", _PUNCT.sub(" ", text)).strip()


def distinct_word_ratio(text: str) -> float:
    words = [w.lower() for w in _WORD.findall(text)]
    return len(set(words)) / len(words) if len(words) >= 8 else 1.0


def drop_reason(moore: str, french: str, cfg: BacktranslateConfig) -> str | None:
    """Why a synthetic pair should not be used, or None if it passes."""
    french = french.strip()
    if not french:
        return "empty"
    if _normalize(french) == _normalize(moore):
        return "copy"
    if sum(c in MOORE_ONLY_LETTERS for c in french) >= 2:
        return "moore_letters"
    ratio = len(french) / max(len(moore), 1)
    if not cfg.min_length_ratio <= ratio <= cfg.max_length_ratio:
        return "length_ratio"
    if distinct_word_ratio(french) < cfg.min_distinct_ratio:
        return "loop"
    return None


def to_pair(row: dict, french: str, cfg: BacktranslateConfig) -> dict:
    moore = row[cfg.text_field]
    return {
        "id": row["id"],
        "french": french.strip(),
        "moore": moore,
        "source": f"bt-{row.get('source') or 'mono'}",
        "original_lang": "mos",
        "doc_id": row.get("doc_id"),
        "reviewed": False,
        "bt_model": cfg.model,
        "drop_reason": drop_reason(moore, french, cfg),
    }


def done_ids(path: Path) -> set[str]:
    if not path.exists():
        return set()
    with path.open(encoding="utf-8") as f:
        return {json.loads(line)["id"] for line in f if line.strip()}


def backtranslate(
    rows: list[dict], translate: TranslateFn, cfg: BacktranslateConfig, path: Path
) -> Counter:
    """Translate rows not yet in `path`, appending pairs batch by batch."""
    already = done_ids(path)
    todo = [r for r in rows if r["id"] not in already]
    print(f"{len(rows)} rows, {len(already)} already translated, {len(todo)} to go")
    path.parent.mkdir(parents=True, exist_ok=True)
    reasons: Counter = Counter()
    start = time.time()
    with path.open("a", encoding="utf-8") as f:
        for i in range(0, len(todo), cfg.batch_size):
            batch = todo[i : i + cfg.batch_size]
            outputs = translate([r[cfg.text_field] for r in batch])
            for row, french in zip(batch, outputs, strict=True):
                pair = to_pair(row, french, cfg)
                reasons[pair["drop_reason"] or "kept"] += 1
                f.write(json.dumps(pair, ensure_ascii=False) + "\n")
            f.flush()
            done = i + len(batch)
            if done % (cfg.batch_size * 20) == 0 or done == len(todo):
                print(f"  {done}/{len(todo)} ({time.time() - start:.0f}s)", flush=True)
    return reasons


def load_rows(cfg: BacktranslateConfig) -> list[dict]:
    from datasets import load_dataset

    ds = load_dataset(
        cfg.dataset, name=cfg.dataset_config, revision=cfg.dataset_revision, split=cfg.split
    )
    if cfg.limit is not None:
        ds = ds.select(range(min(cfg.limit, len(ds))))
    return list(ds)


def model_translate_fn(cfg: BacktranslateConfig) -> TranslateFn:
    from mt_training.inference import load_model, translate_batch

    model, tokenizer = load_model(cfg.model, dtype=cfg.dtype)

    def translate(texts: list[str]) -> list[str]:
        return translate_batch(
            texts,
            model,
            tokenizer,
            cfg.src_lang,
            cfg.tgt_lang,
            cfg.beam_size,
            cfg.no_repeat_ngram_size,
            cfg.max_new_tokens,
        )

    return translate


@draccus.wrap()
def main(cfg: BacktranslateConfig) -> None:
    if not cfg.model:
        raise SystemExit("--model is required (a Mooré -> French model or CT2 directory)")
    path = cfg.output_dir_root / cfg.output
    reasons = backtranslate(load_rows(cfg), model_translate_fn(cfg), cfg, path)
    total = sum(reasons.values())
    print(f"Wrote {total} pairs to {path}")
    for reason, n in reasons.most_common():
        print(f"  {reason:14} {n:>6} ({n / max(total, 1):.1%})")


if __name__ == "__main__":
    main()
