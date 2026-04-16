"""
Backtranslation script using quickmt.

Pipeline: French → English → French
  1. Translate French monolingual text to English  (quickmt/quickmt-fr-en)
  2. Translate that English back to French          (quickmt/quickmt-en-fr)

Produces a dataset with columns:
  - french_original   : the input French sentence
  - english           : intermediate English translation
  - french_synthetic  : round-trip French (backtranslated)
  - source            : "backtranslation"

The synthetic French can be used to augment the French source side of a
French→Mooré training set.

Usage:
    uv run python -m mt_training.backtranslate \
        --dataset_id madoss/fr-mos-final-data \
        --text_column french \
        --output_path data/backtranslated \
        --device cpu
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import cast

from datasets import Dataset, load_dataset
from huggingface_hub import snapshot_download
from transformers import HfArgumentParser
from dotenv import load_dotenv

load_dotenv()
try:
    from quickmt import Translator
except ImportError as e:
    raise ImportError(
        "quickmt is required for backtranslation. Install it with: uv sync --extra backtranslation"
    ) from e

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

QUICKMT_FR_EN = "quickmt/quickmt-fr-en"
QUICKMT_EN_FR = "quickmt/quickmt-en-fr"


@dataclass
class BacktranslationArguments:
    dataset_id: str = field(
        metadata={"help": "HuggingFace dataset ID with French monolingual text"},
    )
    text_column: str = field(
        default="french",
        metadata={"help": "Column name that holds the French source text"},
    )
    output_path: str = field(
        default="data/backtranslated",
        metadata={"help": "Local path to save the backtranslated dataset"},
    )
    split: str = field(
        default="train",
        metadata={"help": "Dataset split to use"},
    )
    max_samples: int = field(
        default=-1,
        metadata={"help": "Maximum number of samples to translate (-1 = all)"},
    )
    beam_size: int = field(
        default=3,
        metadata={"help": "Beam size (1 = fastest, higher = better quality)"},
    )
    device: str = field(
        default="auto",
        metadata={"help": "Device: 'auto', 'cpu', or 'cuda'"},
    )
    push_to_hub: bool = field(
        default=False,
        metadata={"help": "Push the resulting dataset to HuggingFace Hub"},
    )
    hub_dataset_id: str = field(
        default="",
        metadata={"help": "HF Hub dataset repo to push to (required when push_to_hub=True)"},
    )


def load_quickmt(model_id: str, device: str) -> Translator:
    logger.info("Downloading quickmt model: %s", model_id)
    model_path = snapshot_download(model_id, ignore_patterns="eole-model/*")
    return Translator(model_path, device=device)


def main() -> None:
    parser = HfArgumentParser((BacktranslationArguments,))  # type: ignore
    (args,) = cast(
        tuple[BacktranslationArguments],
        parser.parse_args_into_dataclasses(),
    )

    logger.info("Loading dataset: %s (split=%s)", args.dataset_id, args.split)
    raw = load_dataset(args.dataset_id, split=args.split)

    if args.max_samples > 0:
        raw = raw.select(range(min(args.max_samples, len(raw))))  # type: ignore[arg-type]

    french_original: list[str] = raw[args.text_column]  # type: ignore[index]
    logger.info("Loaded %d sentences", len(french_original))

    # Step 1: French → English (full dataset in one pass)
    fr_en = load_quickmt(QUICKMT_FR_EN, args.device)
    logger.info("Translating fr→en (%d sentences)", len(french_original))
    english = cast(list[str], fr_en(french_original, beam_size=args.beam_size))
    del fr_en  # free memory before loading the next model

    # Step 2: English → French (full dataset in one pass)
    en_fr = load_quickmt(QUICKMT_EN_FR, args.device)
    logger.info("Translating en→fr (%d sentences)", len(english))
    french_synthetic = cast(list[str], en_fr(english, beam_size=args.beam_size))
    del en_fr

    dataset = Dataset.from_dict(
        {
            "french_original": french_original,
            "english": english,
            "french_synthetic": french_synthetic,
            "source": ["backtranslation"] * len(french_original),
        }
    )

    logger.info("Saving dataset to %s", args.output_path)
    dataset.save_to_disk(args.output_path)

    if args.push_to_hub:
        if not args.hub_dataset_id:
            raise ValueError("--hub_dataset_id is required when --push_to_hub is set")
        logger.info("Pushing dataset to Hub: %s", args.hub_dataset_id)
        dataset.push_to_hub(args.hub_dataset_id)

    logger.info("Done. %d rows written to %s", len(dataset), args.output_path)


if __name__ == "__main__":
    main()
