from datasets import Dataset, DatasetDict

from mt_training.mix import MixConfig, mix


def parallel_row(i: int, french: str, moore: str) -> dict:
    return {
        "id": f"p-{i}",
        "french": french,
        "moore": moore,
        "source": "news",
        "original_lang": "fra",
        "doc_id": "p",
        "reviewed": True,
        "laser_score": 0.9,
        "comet_qe": 0.7,
        "len_ratio": 0.9,
    }


def bt_row(i: int, french: str, moore: str, **extra) -> dict:
    row = {
        "id": f"hplt-{i}",
        "french": french,
        "moore": moore,
        "source": "bt-wikipedia",
        "original_lang": "mos",
        "doc_id": "d",
        "reviewed": False,
        "bt_model": "m",
        "drop_reason": None,
        "laser_score": 0.8,
        "comet_qe": 0.6,
    }
    return row | extra


PARALLEL = DatasetDict(
    {
        "train": Dataset.from_list([parallel_row(0, "Bonjour.", "Ne y yibeoogo.")]),
        "validation": Dataset.from_list([parallel_row(1, "La pluie tombe.", "Saag ningda.")]),
        "test": Dataset.from_list([parallel_row(2, "Il mange.", "A rɩtame.")]),
    }
)


def test_mix_filters_and_keeps_eval_authentic():
    rows = [
        bt_row(0, "Le chien court.", "Baaga zoeta."),
        bt_row(1, "Oui.", "Baaga zoeta wʋsgo.", drop_reason="length_ratio"),
        bt_row(2, "La pluie tombe !", "Saag ningda wã."),  # French is a validation sentence
        bt_row(3, "Le chat dort.", "Baaga zoeta."),  # same Mooré as a kept pair
        bt_row(4, "Il fait chaud.", "Wĩntoogã tara tʋʋlgo.", comet_qe=0.3),
    ]
    mixed, reasons = mix(PARALLEL, rows, MixConfig())
    assert reasons == {"kept": 2, "rule:length_ratio": 1, "in_parallel": 2}
    assert len(mixed["train"]) == 3
    assert mixed["train"].features == PARALLEL["train"].features
    assert len(mixed["validation"]) == len(mixed["test"]) == 1
    synthetic = [r for r in mixed["train"] if r["source"] == "bt-wikipedia"]
    assert {r["id"] for r in synthetic} == {"hplt-0", "hplt-4"}
    assert not any(r["reviewed"] for r in synthetic)


def test_mix_score_thresholds():
    rows = [
        bt_row(0, "Le chien court.", "Baaga zoeta."),
        bt_row(4, "Il fait chaud.", "Wĩntoogã tara tʋʋlgo.", comet_qe=0.3),
        bt_row(5, "Il pleut fort.", "Saag ningda wʋsgo.", comet_qe=None),
    ]
    _, reasons = mix(PARALLEL, rows, MixConfig(min_comet_qe=0.5))
    assert reasons == {"kept": 1, "comet_qe": 2}
