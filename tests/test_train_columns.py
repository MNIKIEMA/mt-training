import pytest
from datasets import Dataset, DatasetDict

from mt_training.train import DataTrainingArguments, map_columns


def _ds():
    return DatasetDict(
        {
            "train": Dataset.from_dict(
                {"french": ["Bonjour."], "moore": ["Ne y yibeoogo."], "source": ["expert"]}
            )
        }
    )


def test_default_mapping_is_french_to_moore():
    ds = map_columns(_ds(), DataTrainingArguments())["train"][0]
    assert (ds["source"], ds["target"], ds["data_source"]) == (
        "Bonjour.",
        "Ne y yibeoogo.",
        "expert",
    )


def test_swapped_mapping_is_moore_to_french():
    args = DataTrainingArguments(
        source_field="moore", target_field="french", src_lang="mos_Latn", tgt_lang="fra_Latn"
    )
    ds = map_columns(_ds(), args)["train"][0]
    assert (ds["source"], ds["target"], ds["data_source"]) == (
        "Ne y yibeoogo.",
        "Bonjour.",
        "expert",
    )


def test_same_field_twice_is_rejected():
    with pytest.raises(ValueError):
        map_columns(_ds(), DataTrainingArguments(source_field="moore", target_field="moore"))
