"""Add backtranslated pairs to the train split of the authentic parallel data.

    mt-training mix-bt --bt_files '[data/bt/moore-web-mono-v1.1.0-bt-MosFr-v2-scored.jsonl]' \
        --output_dir data/mix/mwp-v1.1.0-bt-mosfr-v2

Backtranslated pairs (`mt_training.backtranslate`, optionally scored with
LASER / COMET-QE by moore-web's `annotate`) are kept when:

- `drop_reason` is empty (the rule checks at translation time passed);
- `comet_qe` / `laser_score` reach `--min_comet_qe` / `--min_laser` when set
  (off by default; a pair without the score is dropped when its threshold is set);
- neither side repeats a sentence of the parallel data, in any split
  (normalized as in `mt_training.overlap`): a validation/test sentence must not
  enter train, and a train sentence needs no synthetic copy.

Validation and test stay authentic. Rows take the parallel schema; `source`
(`bt-wikipedia`, …) marks them as synthetic. The output directory holds
`train.parquet`, `validation.parquet` and `test.parquet`, which
`load_dataset(<dir>)` reads, so `train.py --dataset_id <dir>` works without
the Hub; `--push_to_hub <repo>` uploads a private copy for Modal runs.
"""

import json
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

import draccus
from datasets import Dataset, DatasetDict, concatenate_datasets, load_dataset

from mt_training.overlap import normalize


@dataclass
class MixConfig:
    bt_files: list[Path] = field(default_factory=list, metadata={"help": "Backtranslated JSONL"})
    dataset: str = field(default="madoss/moore-web-parallel", metadata={"help": "Parallel data"})
    dataset_config: str | None = field(default="mos-fra", metadata={"help": "Dataset config"})
    dataset_revision: str | None = field(default="v1.1.0", metadata={"help": "Tag or commit"})
    min_comet_qe: float | None = field(default=None, metadata={"help": "Keep comet_qe >= this"})
    min_laser: float | None = field(default=None, metadata={"help": "Keep laser_score >= this"})
    output_dir: Path = field(default=Path("data/mix/mix"), metadata={"help": "Parquet output"})
    push_to_hub: str | None = field(default=None, metadata={"help": "Private Hub dataset repo"})


def load_jsonl(paths: list[Path]) -> list[dict]:
    rows = []
    for path in paths:
        with path.open(encoding="utf-8") as f:
            rows += [json.loads(line) for line in f if line.strip()]
    return rows


def reject_reason(row: dict, seen: set[str], cfg: MixConfig) -> str | None:
    """Why a backtranslated pair is left out, or None to keep it."""
    if row.get("drop_reason"):
        return f"rule:{row['drop_reason']}"
    for name, key, minimum in (
        ("comet_qe", "comet_qe", cfg.min_comet_qe),
        ("laser", "laser_score", cfg.min_laser),
    ):
        if minimum is not None and (row.get(key) is None or row[key] < minimum):
            return name
    if normalize(row["moore"]) in seen or normalize(row["french"]) in seen:
        return "in_parallel"
    return None


def select_pairs(
    rows: list[dict], parallel: DatasetDict, cfg: MixConfig
) -> tuple[list[dict], Counter]:
    seen = {
        normalize(t) for split in parallel.values() for c in ("french", "moore") for t in split[c]
    }
    kept, reasons = [], Counter()
    for row in rows:
        reason = reject_reason(row, seen, cfg)
        reasons[reason or "kept"] += 1
        if reason is None:
            kept.append(row)
            # Two synthetic pairs with the same Mooré sentence add nothing.
            seen.add(normalize(row["moore"]))
    return kept, reasons


def to_parallel_schema(rows: list[dict], features) -> Dataset:
    columns = {name: [row.get(name) for row in rows] for name in features}
    columns["reviewed"] = [False] * len(rows)
    return Dataset.from_dict(columns, features=features)


def mix(parallel: DatasetDict, rows: list[dict], cfg: MixConfig) -> tuple[DatasetDict, Counter]:
    kept, reasons = select_pairs(rows, parallel, cfg)
    train = parallel["train"]
    synthetic = to_parallel_schema(kept, train.features)
    mixed = DatasetDict(parallel)
    mixed["train"] = concatenate_datasets([train, synthetic]).shuffle(seed=42)
    return mixed, reasons


@draccus.wrap()
def main(cfg: MixConfig) -> None:
    if not cfg.bt_files:
        raise SystemExit("--bt_files is required")
    parallel = load_dataset(cfg.dataset, name=cfg.dataset_config, revision=cfg.dataset_revision)
    assert isinstance(parallel, DatasetDict)
    rows = load_jsonl(cfg.bt_files)
    mixed, reasons = mix(parallel, rows, cfg)

    print(f"{len(rows)} backtranslated pairs from {', '.join(map(str, cfg.bt_files))}")
    for reason, n in reasons.most_common():
        print(f"  {reason:18} {n:>6}")
    n_train = len(parallel["train"])
    print(f"train: {n_train} authentic + {reasons['kept']} synthetic = {len(mixed['train'])}")

    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    for split, ds in mixed.items():
        ds.to_parquet(cfg.output_dir / f"{split}.parquet")
    print(f"Wrote {cfg.output_dir}/{{{','.join(mixed)}}}.parquet")
    if cfg.push_to_hub:
        mixed.push_to_hub(
            cfg.push_to_hub, config_name=cfg.dataset_config or "default", private=True
        )
        print(f"Pushed to {cfg.push_to_hub} (private)")


if __name__ == "__main__":
    main()
