"""
Convert a HuggingFace seq2seq model to CTranslate2 format.

Usage:
    mt-convert-ct2 --model madoss/nllb-200-finetuned-600-FRA-MOS --output_dir ./ct2-model
    mt-convert-ct2 --model ./local-model --output_dir ./ct2-model --quantization float16
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal
from dotenv import load_dotenv
import draccus

load_dotenv()


@dataclass
class ConvertConfig:
    model: str = field(
        metadata={"help": "HuggingFace model name or local path to convert"},
    )
    output_dir: str = field(
        metadata={"help": "Directory where the CTranslate2 model will be saved"},
    )
    quantization: Literal[
        "int8", "int8_float16", "int8_bfloat16", "float16", "bfloat16", "float32"
    ] = field(
        default="int8",
        metadata={"help": "Weight quantization type (default: int8)"},
    )
    force: bool = field(
        default=False,
        metadata={"help": "Overwrite output_dir if it already exists"},
    )


@draccus.wrap()
def main(cfg: ConvertConfig) -> None:
    try:
        import ctranslate2
    except ImportError as e:
        raise ImportError(
            "ctranslate2 is required for model conversion. "
            "Install project dependencies with: uv sync"
        ) from e

    output_dir = Path(cfg.output_dir)
    if output_dir.exists() and not cfg.force:
        raise FileExistsError(f"{output_dir} already exists. Use --force to overwrite.")

    print(f"Converting: {cfg.model}")
    print(f"Output    : {output_dir}")
    print(f"Quantize  : {cfg.quantization}")

    converter = ctranslate2.converters.TransformersConverter(
        cfg.model,
        low_cpu_mem_usage=True,
    )
    output_path = converter.convert(
        str(output_dir),
        quantization=cfg.quantization,
        force=cfg.force,
    )
    print(f"Done — model saved to {output_path}")


if __name__ == "__main__":
    main()
