import csv

import pytest

from mt_training.compare import CompareConfig, compare, length_buckets


def _write(path, rows):
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["source", "reference", "hypothesis"])
        writer.writeheader()
        writer.writerows(rows)


def test_length_buckets_are_equal_size_and_cover_everything():
    sources = ["a" * n for n in (5, 1, 4, 2, 3, 6, 8, 7)]
    buckets = length_buckets(sources, 4)
    names = [name for name, _ in buckets]
    assert names == ["Q1 (1-2)", "Q2 (3-4)", "Q3 (5-6)", "Q4 (7-8)", "All"]
    assert [len(idx) for _, idx in buckets] == [2, 2, 2, 2, 8]


def test_candidate_matching_references_beats_baseline(tmp_path):
    refs = [f"Tõnd na n kẽnga yiri {i} ye." for i in range(40)]
    sources = [f"Nous irons à la maison {i}." + " x" * i for i in range(40)]
    _write(
        tmp_path / "base.csv",
        [
            {"source": s, "reference": r, "hypothesis": "Ned ka be ye."}
            for s, r in zip(sources, refs)
        ],
    )
    _write(
        tmp_path / "new.csv",
        [{"source": s, "reference": r, "hypothesis": r} for s, r in zip(sources, refs)],
    )
    rows = compare(
        CompareConfig(tmp_path / "base.csv", tmp_path / "new.csv", buckets=2, resamples=50)
    )
    overall = rows[-1]
    assert overall["bucket"] == "All" and overall["n"] == 40
    assert overall["chrf++_candidate"] == pytest.approx(100.0)
    assert overall["ci95_low"] > 0
    assert overall["len_ratio_candidate"] == pytest.approx(1.0)


def test_rejects_files_for_different_examples(tmp_path):
    _write(tmp_path / "a.csv", [{"source": "Un.", "reference": "A.", "hypothesis": "A."}])
    _write(tmp_path / "b.csv", [{"source": "Deux.", "reference": "A.", "hypothesis": "A."}])
    with pytest.raises(ValueError):
        compare(CompareConfig(tmp_path / "a.csv", tmp_path / "b.csv"))
