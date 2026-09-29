from datasets import Dataset

from mt_training.overlap import OverlapConfig, find_overlap, index_training, normalize, word_bucket


def test_normalize_ignores_case_punctuation_and_spacing_but_keeps_accents():
    assert normalize("  AU TITRE DU MINISTÈRE…  Le Conseil a adopté ! ") == normalize(
        "au titre du ministère le conseil a adopté"
    )
    assert normalize("-la délimitation des terrains") == "la délimitation des terrains"
    assert normalize("« Non ! »") == "non"
    assert normalize("ministère") != normalize("ministere")


def test_find_overlap_reports_first_split_and_source_with_word_count():
    train = {
        "train": Dataset.from_dict(
            {
                "french": ["La délimitation des terrains.", "froid"],
                "source": ["conseils", "lexicon"],
            }
        ),
        "test": Dataset.from_dict({"french": ["Autre phrase."], "source": ["sida"]}),
    }
    index = index_training(train, "french", "source")
    rows = [
        {"id": 0, "source_text": "-la délimitation des terrains", "domain": "administrative"},
        {"id": 1, "source_text": "Froid", "domain": "unknown"},
        {"id": 2, "source_text": "Jamais vu.", "domain": "daily_life"},
        {"id": 3, "source_text": "autre phrase", "domain": "health"},
    ]
    assert find_overlap(rows, index, OverlapConfig()) == [
        {"id": 0, "domain": "administrative", "seen_in": "train:conseils", "source_words": 4},
        {"id": 1, "domain": "unknown", "seen_in": "train:lexicon", "source_words": 1},
        {"id": 3, "domain": "health", "seen_in": "test:sida", "source_words": 2},
    ]


def test_word_buckets():
    assert [word_bucket(n) for n in (1, 2, 3, 5, 6, 10, 11, 80)] == [
        "1-2 words",
        "1-2 words",
        "3-5 words",
        "3-5 words",
        "6-10 words",
        "6-10 words",
        "11+ words",
        "11+ words",
    ]
