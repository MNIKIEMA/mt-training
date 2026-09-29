"""
Find benchmark sources that also appear in a training dataset.

Both sides are normalized the same way (Unicode NFKC, lowercase, punctuation
removed, spaces collapsed; accents kept), and a benchmark row matches when its
normalized source equals a normalized training source. Only exact matches are
found: a sentence that differs by a word, or is split differently, is not
counted, so the result is a lower bound on the overlap.

Short matches are usually ordinary words or fragments that any parallel corpus
contains (dictionary headwords, "Non !"); the `source_words` column lets you
keep only full sentences (e.g. 6+ words), which point to shared documents.

Usage:
    python -m mt_training.overlap --output overlap.csv
    python -m mt_training.overlap --benchmark burkimbia/mt-benchmark-public \\
        --train_dataset madoss/moore-web-parallel --train_config mos-fra \\
        --train_revision v1.0.0
"""

import csv
import re
import unicodedata
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import cast

import draccus
from datasets import Dataset, DatasetDict, load_dataset

_PUNCT = re.compile(r"[^\w\s]")
_SPACES = re.compile(r"\s+")
WORD_BUCKETS = ((1, 2), (3, 5), (6, 10), (11, None))


@dataclass
class OverlapConfig:
    benchmark: str = field(
        default="burkimbia/mt-benchmark-public", metadata={"help": "Benchmark Hub dataset"}
    )
    benchmark_split: str = field(default="train", metadata={"help": "Benchmark split"})
    benchmark_id_field: str = field(default="id", metadata={"help": "Benchmark id column"})
    benchmark_text_field: str = field(
        default="source_text", metadata={"help": "Benchmark source text column"}
    )
    group_field: str = field(
        default="domain", metadata={"help": "Benchmark column to group counts by, if present"}
    )
    train_dataset: str = field(
        default="madoss/moore-web-parallel", metadata={"help": "Training Hub dataset"}
    )
    train_config: str | None = field(default="mos-fra", metadata={"help": "Training config"})
    train_revision: str | None = field(default="v1.0.0", metadata={"help": "Training revision"})
    train_text_field: str = field(default="french", metadata={"help": "Training source column"})
    train_source_field: str = field(
        default="source", metadata={"help": "Training column naming the data source"}
    )
    output: Path = field(default=Path("overlap.csv"), metadata={"help": "Matches as CSV"})


def normalize(text: str) -> str:
    text = unicodedata.normalize("NFKC", text).lower()
    return _SPACES.sub(" ", _PUNCT.sub(" ", text)).strip()


def index_training(splits: dict[str, Dataset], text_field: str, source_field: str) -> dict:
    """Normalized training text -> "split:source" of its first occurrence."""
    index: dict[str, str] = {}
    for split, ds in splits.items():
        sources = ds[source_field] if source_field in ds.column_names else [""] * len(ds)
        for text, source in zip(ds[text_field], sources, strict=True):
            index.setdefault(normalize(text), f"{split}:{source}")
    return index


def find_overlap(rows: list[dict], index: dict[str, str], cfg: OverlapConfig) -> list[dict]:
    matches = []
    for row in rows:
        text = row[cfg.benchmark_text_field]
        seen_in = index.get(normalize(text))
        if seen_in is None:
            continue
        match = {"id": row[cfg.benchmark_id_field]}
        if cfg.group_field in row:
            match[cfg.group_field] = row[cfg.group_field]
        match |= {"seen_in": seen_in, "source_words": len(text.split())}
        matches.append(match)
    return matches


def word_bucket(n: int) -> str:
    for lo, hi in WORD_BUCKETS:
        if hi is None or n <= hi:
            return f"{lo}+ words" if hi is None else f"{lo}-{hi} words"
    raise AssertionError


def print_summary(matches: list[dict], total: int, group_field: str) -> None:
    print(f"Exact matches: {len(matches)} of {total} benchmark rows")
    by_bucket: dict[str, list[dict]] = {}
    for m in matches:
        by_bucket.setdefault(word_bucket(m["source_words"]), []).append(m)
    for lo, hi in WORD_BUCKETS:
        name = f"{lo}+ words" if hi is None else f"{lo}-{hi} words"
        group = by_bucket.get(name, [])
        groups = dict(Counter(m.get(group_field, "-") for m in group).most_common())
        sources = dict(Counter(m["seen_in"].split(":", 1)[1] for m in group).most_common())
        print(f"  {name:10} {len(group):>5}  {group_field}: {groups}  seen in: {sources}")


@draccus.wrap()
def main(cfg: OverlapConfig) -> None:
    benchmark = cast(Dataset, load_dataset(cfg.benchmark, split=cfg.benchmark_split))
    train = cast(
        DatasetDict,
        load_dataset(cfg.train_dataset, name=cfg.train_config, revision=cfg.train_revision),
    )
    index = index_training(dict(train), cfg.train_text_field, cfg.train_source_field)
    matches = find_overlap(list(benchmark), index, cfg)
    print_summary(matches, len(benchmark), cfg.group_field)
    if matches:
        with cfg.output.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(matches[0]))
            writer.writeheader()
            writer.writerows(matches)
    print(f"Saved {len(matches)} matches to {cfg.output}")


if __name__ == "__main__":
    main()
