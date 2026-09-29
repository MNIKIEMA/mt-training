"""
Text normalization for the NLLB tokenizer.

NLLB's vocabulary has no typographic apostrophe (’) and no guillemets (« »):
the tokenizer turns them into <unk>, and decoding drops them ("C’est l’état"
-> "Cest létat"). A model trained on such text learns to emit <unk> where they
belong. So every text is normalized the same way right before tokenization, in
training and at inference: typographic apostrophes and quotes become the ASCII
ones NLLB knows, and « mot » becomes "mot". The datasets keep their typography.

Check a dataset for characters that still become <unk>:
    python -m mt_training.text --dataset madoss/moore-web-parallel --config mos-fra --revision v1.0.0
"""

import re
import unicodedata
from collections import Counter
from dataclasses import dataclass, field

import draccus

_TRANSLATE = str.maketrans(
    {
        "’": "'",
        "‘": "'",
        "ʼ": "'",
        "“": '"',
        "”": '"',
        "„": '"',
        "—": "-",
        "–": "-",
        "\u0342": "\u0303",  # Greek perispomeni used as a tilde -> combining tilde
        "ɭ": "Ɩ",  # used as capital ɩ in all-caps conseils headings ("Tɭ" = "TƖ")
    }
)
# Spaces inside guillemets are French typography; straight quotes hug the text.
_OPEN_GUILLEMET = re.compile(r"«\s*")
_CLOSE_GUILLEMET = re.compile(r"\s*»")


def normalize_for_nllb(text: str) -> str:
    """Map characters NLLB's tokenizer can't represent to ones it can."""
    text = _OPEN_GUILLEMET.sub('"', text)
    text = _CLOSE_GUILLEMET.sub('"', text)
    text = text.translate(_TRANSLATE)
    return unicodedata.normalize("NFC", text)


def unknown_characters(texts: list[str], tokenizer) -> Counter:
    """Characters that still become <unk> after normalization, with row counts."""
    unk = tokenizer.unk_token_id
    found: Counter = Counter()
    for text in texts:
        for char in set(normalize_for_nllb(text)):
            if char.isspace():
                continue
            if unk in tokenizer(f"a{char}b", add_special_tokens=False)["input_ids"]:
                found[char] += 1
    return found


@dataclass
class CheckConfig:
    dataset: str = field(default="madoss/moore-web-parallel", metadata={"help": "Hub dataset"})
    config: str | None = field(default="mos-fra", metadata={"help": "Dataset config"})
    revision: str | None = field(default="v1.0.0", metadata={"help": "Dataset revision"})
    fields: list[str] = field(
        default_factory=lambda: ["french", "moore"], metadata={"help": "Text columns"}
    )
    tokenizer: str = field(
        default="facebook/nllb-200-distilled-600M", metadata={"help": "Tokenizer"}
    )


@draccus.wrap()
def main(cfg: CheckConfig) -> None:
    from datasets import load_dataset
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(cfg.tokenizer)
    ds = load_dataset(cfg.dataset, name=cfg.config, revision=cfg.revision)
    for column in cfg.fields:
        texts = [t for split in ds.values() for t in split[column]]
        found = unknown_characters(texts, tokenizer)
        print(
            f"{column}: {len(texts)} rows, characters still <unk> after normalization: "
            f"{ {c: n for c, n in found.most_common()} or 'none' }"
        )


if __name__ == "__main__":
    main()
