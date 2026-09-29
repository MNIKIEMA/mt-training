import json

from mt_training.backtranslate import BacktranslateConfig, backtranslate, drop_reason

CFG = BacktranslateConfig(model="fake")


def test_drop_reasons():
    mos = "Turkmen haly yaa buud a ye sẽn yaa ne nug tʋʋma."
    assert drop_reason(mos, "Le tapis turkmène est un type fait à la main.", CFG) is None
    assert drop_reason(mos, "  ", CFG) == "empty"
    assert drop_reason(mos, "Turkmen haly yaa buud a ye sẽn yaa ne nug tʋʋma !", CFG) == "copy"
    assert drop_reason(mos, "Le haly yaa buud sẽn tʋʋma fait main.", CFG) == "moore_letters"
    assert drop_reason(mos, "Oui.", CFG) == "length_ratio"
    loop = "Le tapis est le tapis qui est le tapis qui est le tapis du tapis."
    assert drop_reason(mos, loop, CFG) == "loop"


def test_backtranslate_writes_pairs_and_resumes(tmp_path):
    rows = [
        {
            "id": f"hplt-{i}",
            "text": f"Tõnd na n kẽnga yiri {i}.",
            "source": "wikipedia",
            "doc_id": "d",
        }
        for i in range(5)
    ]
    calls = []

    def translate(texts):
        calls.append(len(texts))
        return [t.replace("Tõnd na n kẽnga yiri", "Nous irons à la maison") for t in texts]

    out = tmp_path / "bt.jsonl"
    cfg = BacktranslateConfig(model="fake", batch_size=2)
    first = backtranslate(rows[:3], translate, cfg, out)
    assert first == {"kept": 3} and calls == [2, 1]
    second = backtranslate(rows, translate, cfg, out)  # resumes: only the 2 new rows
    assert second == {"kept": 2} and calls == [2, 1, 2]
    pairs = [json.loads(line) for line in out.read_text(encoding="utf-8").splitlines()]
    assert [p["id"] for p in pairs] == [r["id"] for r in rows]
    p = pairs[0]
    assert (p["french"], p["moore"]) == ("Nous irons à la maison 0.", "Tõnd na n kẽnga yiri 0.")
    assert (p["source"], p["original_lang"], p["reviewed"], p["drop_reason"]) == (
        "bt-wikipedia",
        "mos",
        False,
        None,
    )
