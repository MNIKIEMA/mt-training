"""
Compare two models on the same evaluation set, overall and by source length.

Reads two prediction files written by `mt_training.eval --output` (csv or jsonl
with `source`, `reference`, `hypothesis`) for the same examples in the same
order. Reports chrF++ and BLEU per source-length bucket, with a paired
bootstrap 95% confidence interval for the chrF++ difference (candidate -
baseline), and for each model the output/reference length ratio, the share
of outputs longer than `--long_ratio` times their reference, and the mean
share of repeated words in an output (1 - distinct/total words), which flag
a chrF++ gain that comes from longer or looping output rather than better
translations (chrF++ weighs recall twice as much as precision).

Usage:
    python -m mt_training.eval --model base-ct2 --output base.csv
    python -m mt_training.eval --model new-ct2 --output new.csv
    python -m mt_training.compare --baseline base.csv --candidate new.csv
"""

import csv
import json
from dataclasses import dataclass, field
from pathlib import Path

import draccus
import numpy as np
from sacrebleu.metrics import BLEU, CHRF


@dataclass
class CompareConfig:
    baseline: Path = field(default=Path("base.csv"), metadata={"help": "Baseline predictions"})
    candidate: Path = field(default=Path("new.csv"), metadata={"help": "Candidate predictions"})
    buckets: int = field(
        default=4, metadata={"help": "Number of equal-size buckets by source length (chars)"}
    )
    resamples: int = field(default=1000, metadata={"help": "Paired bootstrap resamples"})
    seed: int = field(default=0, metadata={"help": "Bootstrap random seed"})
    long_ratio: float = field(
        default=1.5, metadata={"help": "An output this many times its reference length is long"}
    )
    output: Path | None = field(default=None, metadata={"help": "Also write the rows as JSON"})


def load_predictions(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8") as f:
        if path.suffix == ".jsonl":
            return [json.loads(line) for line in f if line.strip()]
        return list(csv.DictReader(f))


def length_buckets(sources: list[str], n: int) -> list[tuple[str, np.ndarray]]:
    """Equal-size buckets of example indices by source length, then all examples."""
    lengths = np.array([len(s) for s in sources])
    order = np.argsort(lengths, kind="stable")
    buckets = []
    for i, idx in enumerate(np.array_split(order, n), start=1):
        lo, hi = lengths[idx].min(), lengths[idx].max()
        buckets.append((f"Q{i} ({lo}-{hi})", idx))
    buckets.append(("All", np.arange(len(sources))))
    return buckets


def sentence_stats(metric, hypotheses: list[str], references: list[str]) -> np.ndarray:
    # Per-sentence sufficient statistics; corpus scores are computed from their sums,
    # as in sacrebleu's own paired bootstrap (sacrebleu.significance).
    return np.array(metric._extract_corpus_statistics(hypotheses, [references]))


def corpus_score(metric, stats: np.ndarray) -> float:
    return metric._compute_score_from_stats(stats.sum(0)).score


def repeated_word_share(text: str) -> float:
    """1 - distinct/total words: 0 for no repetition, near 1 for a loop."""
    words = text.split()
    return 1 - len(set(words)) / len(words) if words else 0.0


def compare(cfg: CompareConfig) -> list[dict]:
    base, cand = load_predictions(cfg.baseline), load_predictions(cfg.candidate)
    if [r["source"] for r in base] != [r["source"] for r in cand]:
        raise ValueError("The two files must contain the same sources in the same order")
    if [r["reference"] for r in base] != [r["reference"] for r in cand]:
        raise ValueError("The two files must contain the same references")

    sources = [r["source"] for r in base]
    refs = [r["reference"] for r in base]
    hyp_a, hyp_b = [r["hypothesis"] for r in base], [r["hypothesis"] for r in cand]

    chrf, bleu = CHRF(word_order=2), BLEU()
    chrf_a, chrf_b = sentence_stats(chrf, hyp_a, refs), sentence_stats(chrf, hyp_b, refs)
    bleu_a, bleu_b = sentence_stats(bleu, hyp_a, refs), sentence_stats(bleu, hyp_b, refs)
    ref_len = np.array([len(r) for r in refs])
    len_a, len_b = np.array([len(h) for h in hyp_a]), np.array([len(h) for h in hyp_b])
    long_a, long_b = len_a > cfg.long_ratio * ref_len, len_b > cfg.long_ratio * ref_len
    rep_a = np.array([repeated_word_share(h) for h in hyp_a])
    rep_b = np.array([repeated_word_share(h) for h in hyp_b])

    rng = np.random.default_rng(cfg.seed)
    rows = []
    for name, idx in length_buckets(sources, cfg.buckets):
        samples = idx[rng.integers(0, len(idx), size=(cfg.resamples, len(idx)))]
        diffs = np.sort(
            [corpus_score(chrf, chrf_b[s]) - corpus_score(chrf, chrf_a[s]) for s in samples]
        )
        a, b = corpus_score(chrf, chrf_a[idx]), corpus_score(chrf, chrf_b[idx])
        rows.append(
            {
                "bucket": name,
                "n": len(idx),
                "chrf++_baseline": round(a, 2),
                "chrf++_candidate": round(b, 2),
                "chrf++_gain": round(b - a, 2),
                "ci95_low": round(float(np.percentile(diffs, 2.5)), 2),
                "ci95_high": round(float(np.percentile(diffs, 97.5)), 2),
                "bleu_baseline": round(corpus_score(bleu, bleu_a[idx]), 2),
                "bleu_candidate": round(corpus_score(bleu, bleu_b[idx]), 2),
                "len_ratio_baseline": round(len_a[idx].sum() / ref_len[idx].sum(), 2),
                "len_ratio_candidate": round(len_b[idx].sum() / ref_len[idx].sum(), 2),
                "long_share_baseline": round(float(long_a[idx].mean()), 3),
                "long_share_candidate": round(float(long_b[idx].mean()), 3),
                "repeat_share_baseline": round(float(rep_a[idx].mean()), 3),
                "repeat_share_candidate": round(float(rep_b[idx].mean()), 3),
            }
        )
    return rows


def print_table(rows: list[dict]) -> None:
    print(
        f"{'bucket':14} {'n':>5} | {'chrF++ base':>11} {'cand':>6} {'gain':>6} {'95% CI':>16}"
        f" | {'BLEU base':>9} {'cand':>5} | {'len base':>8} {'cand':>5}"
        f" | {'long base':>9} {'cand':>5} | {'rep base':>8} {'cand':>5}"
    )
    for r in rows:
        ci = f"[{r['ci95_low']:+.2f}, {r['ci95_high']:+.2f}]"
        print(
            f"{r['bucket']:14} {r['n']:>5} | {r['chrf++_baseline']:>11.2f}"
            f" {r['chrf++_candidate']:>6.2f} {r['chrf++_gain']:>+6.2f} {ci:>16}"
            f" | {r['bleu_baseline']:>9.2f} {r['bleu_candidate']:>5.2f}"
            f" | {r['len_ratio_baseline']:>8.2f} {r['len_ratio_candidate']:>5.2f}"
            f" | {r['long_share_baseline']:>9.1%} {r['long_share_candidate']:>5.1%}"
            f" | {r['repeat_share_baseline']:>8.3f} {r['repeat_share_candidate']:>5.3f}"
        )


@draccus.wrap()
def main(cfg: CompareConfig) -> None:
    rows = compare(cfg)
    print_table(rows)
    if cfg.output is not None:
        cfg.output.write_text(json.dumps(rows, indent=2) + "\n", encoding="utf-8")
        print(f"Saved to {cfg.output}")


if __name__ == "__main__":
    main()
