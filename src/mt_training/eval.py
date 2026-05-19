"""
Evaluation script: load a dataset, run inference, compute BLEU/chrF++.

Usage:
    # HuggingFace hub dataset
    python -m mt_training.eval --dataset madoss/fr-mos-final-data --src_field french --ref_field moore

    # S3/Tigris Arrow dataset (requires AWS env vars)
    python -m mt_training.eval --dataset s3://burkimbia-store/evaluation/references/MT/dataset

    # From a YAML config file
    python -m mt_training.eval --config eval_config.yaml

Required env vars for S3 datasets (or .env file):
    AWS_ACCESS_KEY_ID
    AWS_SECRET_ACCESS_KEY
    AWS_ENDPOINT_URL_S3       (e.g. https://fly.storage.tigris.dev)
    AWS_REGION                (e.g. auto)
"""

import csv
import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, cast

import draccus
import evaluate
import pandas as pd
import torch
from datasets import Dataset, DownloadConfig, load_dataset, load_from_disk
from dotenv import load_dotenv
from tqdm import tqdm

from mt_training.inference import (
    DEFAULT_MODEL,
    SRC_LANG,
    TGT_LANG,
    load_model,
    translate_batch,
)

load_dotenv()

DEFAULT_DATASET = "s3://burkimbia-store/evaluation/references/MT/dataset"
DEFAULT_SRC_FIELD = "src"
DEFAULT_REF_FIELD = "reference_translation"


@dataclass
class EvalConfig:
    model: str = field(default=DEFAULT_MODEL, metadata={"help": "Model name or local path"})
    dataset: str = field(
        default=DEFAULT_DATASET,
        metadata={"help": "HF hub ID, local path, or s3:// URI"},
    )
    src_field: str = field(
        default=DEFAULT_SRC_FIELD,
        metadata={"help": "Dataset column for source sentences"},
    )
    ref_field: str = field(
        default=DEFAULT_REF_FIELD,
        metadata={"help": "Dataset column for reference translations"},
    )
    src_lang: str = field(default=SRC_LANG, metadata={"help": "Source language code (NLLB format)"})
    tgt_lang: str = field(default=TGT_LANG, metadata={"help": "Target language code (NLLB format)"})
    split: str | None = field(
        default=None,
        metadata={"help": "Dataset split to evaluate on (required for HF hub datasets)"},
    )
    batch_size: int = field(default=16, metadata={"help": "Translation batch size"})
    beam_size: int = field(default=4, metadata={"help": "Beam search width (1 = greedy)"})
    max_new_tokens: int = field(
        default=128, metadata={"help": "Maximum number of tokens to generate"}
    )
    no_repeat_ngram_size: int = field(
        default=3, metadata={"help": "Block repeated n-grams of this size (0 = disabled)"}
    )
    limit: int | None = field(default=None, metadata={"help": "Evaluate on first N examples only"})
    show_samples: int = field(default=5, metadata={"help": "Print N sample translations"})
    output: Path | str = field(
        default="evaluations.csv",
        metadata={"help": "Save predictions to this file (.csv or .jsonl)"},
    )
    output_format: Literal["csv", "jsonl"] = field(
        default="csv", metadata={"help": "Output format when --output is set"}
    )


def build_storage_options() -> dict:
    key = os.environ.get("AWS_ACCESS_KEY_ID")
    secret = os.environ.get("AWS_SECRET_ACCESS_KEY")
    endpoint = os.environ.get("AWS_ENDPOINT_URL_S3", "https://fly.storage.tigris.dev")
    region = os.environ.get("AWS_REGION", "auto")

    if not key or not secret:
        raise EnvironmentError(
            "AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY must be set (directly or via a .env file)."
        )

    return {
        "key": key,
        "secret": secret,
        "endpoint_url": endpoint,
        "client_kwargs": {
            "endpoint_url": endpoint,
            "region_name": region,
        },
    }


FLORES_PLUS = "openlanguagedata/flores_plus"
FLORES_DEFAULT_SPLIT = "devtest"


def _load_flores_plus(src_lang: str, tgt_lang: str, split: str) -> Dataset:
    """Load flores_plus by merging two language configs into src/ref columns."""
    hf_token = os.environ.get("HF_TOKEN")
    dl_cfg = DownloadConfig(token=hf_token)

    df_src = cast(
        pd.DataFrame,
        load_dataset(FLORES_PLUS, name=src_lang, split=split, download_config=dl_cfg).to_pandas(),  # type: ignore[union-attr]
    ).rename(columns={"text": src_lang})
    df_tgt = cast(
        pd.DataFrame,
        load_dataset(FLORES_PLUS, name=tgt_lang, split=split, download_config=dl_cfg).to_pandas(),  # type: ignore[union-attr]
    ).rename(columns={"text": tgt_lang})

    df = pd.merge(df_src, df_tgt, on="id", how="inner")
    return Dataset.from_dict(
        {"src": df[src_lang].tolist(), "reference_translation": df[tgt_lang].tolist()}
    )


def load_eval_dataset(cfg: EvalConfig) -> Dataset:

    if cfg.dataset.startswith("s3://"):
        ds = cast(
            Dataset,
            load_from_disk(cfg.dataset, storage_options=build_storage_options()),
        )
    elif cfg.dataset == FLORES_PLUS:
        split = cfg.split or FLORES_DEFAULT_SPLIT
        print(f"Loading flores_plus ({cfg.src_lang} → {cfg.tgt_lang}, split={split})")
        ds = _load_flores_plus(cfg.src_lang, cfg.tgt_lang, split)
    else:
        if cfg.split is None:
            raise ValueError("--split is required for HuggingFace hub datasets")
        ds = cast(Dataset, load_dataset(cfg.dataset, split=cfg.split))

    print(f"Loaded dataset — {len(ds)} examples, columns: {ds.column_names}")

    for col in (cfg.src_field, cfg.ref_field):
        if col not in ds.column_names:
            raise ValueError(f"Column '{col}' not found. Available: {ds.column_names}")

    if cfg.limit is not None:
        ds = ds.select(range(min(cfg.limit, len(ds))))
        print(f"Using first {len(ds)} examples (--limit {cfg.limit})")

    return ds


def save_predictions(
    path: Path,
    fmt: Literal["csv", "jsonl"],
    sources: list[str],
    references: list[str],
    hypotheses: list[str],
) -> None:
    rows = [
        {"source": src, "reference": ref, "hypothesis": hyp}
        for src, ref, hyp in zip(sources, references, hypotheses, strict=True)
    ]
    if fmt == "csv":
        with path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["source", "reference", "hypothesis"])
            writer.writeheader()
            writer.writerows(rows)
    else:
        with path.open("w", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"Predictions saved to {path}")


def run_evaluation(
    cfg: EvalConfig,
) -> tuple[dict[str, float], list[str], list[str], list[str]]:
    dataset = load_eval_dataset(cfg)

    sources: list[str] = dataset[cfg.src_field]
    references: list[str] = dataset[cfg.ref_field]

    print(f"\nLoading model: {cfg.model}")
    model, tokenizer = load_model(cfg.model)
    device = (
        next(iter(model.parameters())).device
        if hasattr(model, "parameters")
        else torch.device("cpu")
    )
    print(f"Model loaded on: {device}")

    bos_id = tokenizer.convert_tokens_to_ids(cfg.tgt_lang)
    if bos_id == tokenizer.unk_token_id:
        raise ValueError(
            f"'{cfg.tgt_lang}' not found in tokenizer vocabulary — "
            "check tgt_lang or use tokenizer.lang_code_to_id directly."
        )

    hypotheses: list[str] = []
    for i in tqdm(range(0, len(sources), cfg.batch_size), desc="Translating", unit="batch"):
        batch = sources[i : i + cfg.batch_size]
        hypotheses.extend(
            translate_batch(
                batch,
                model,
                tokenizer,
                cfg.src_lang,
                cfg.tgt_lang,
                cfg.beam_size,
                cfg.no_repeat_ngram_size,
                cfg.max_new_tokens,
            )
        )

    refs_wrapped = [[r] for r in references]

    bleu_metric = evaluate.load("sacrebleu")
    chrf_metric = evaluate.load("chrf")

    bleu_result = bleu_metric.compute(predictions=hypotheses, references=refs_wrapped) or {}
    chrf_result = (
        chrf_metric.compute(predictions=hypotheses, references=refs_wrapped, word_order=2) or {}
    )

    metrics = {
        "bleu": round(bleu_result.get("score", 0.0), 2),
        "chrf++": round(chrf_result.get("score", 0.0), 2),
        "num_examples": len(hypotheses),
    }
    return metrics, sources, hypotheses, references


@draccus.wrap()
def main(cfg: EvalConfig) -> None:
    metrics, sources, hypotheses, references = run_evaluation(cfg)

    print("\n" + "=" * 50)
    print("EVALUATION RESULTS")
    print("=" * 50)
    print(f"  Model       : {cfg.model}")
    print(f"  Dataset     : {cfg.dataset}")
    print(f"  Examples    : {metrics['num_examples']}")
    print(f"  BLEU        : {metrics['bleu']}")
    print(f"  chrF++      : {metrics['chrf++']}")
    print("=" * 50)

    if cfg.output is not None:
        output = Path(cfg.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        save_predictions(output, cfg.output_format, sources, references, hypotheses)

    if cfg.show_samples > 0:
        print(f"\nSample translations (first {cfg.show_samples}):")
        for i in range(min(cfg.show_samples, len(hypotheses))):
            print(f"\n[{i + 1}] Reference : {references[i]}")
            print(f"    Hypothesis: {hypotheses[i]}")


if __name__ == "__main__":
    main()
