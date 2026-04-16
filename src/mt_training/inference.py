import argparse
from pathlib import Path
from typing import Protocol, cast

import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, PreTrainedTokenizerBase


class Seq2SeqModel(Protocol):
    device: torch.device

    def generate(self, **kwargs: object) -> torch.LongTensor: ...
    def eval(self) -> "Seq2SeqModel": ...


DEFAULT_MODEL = "madoss/nllb-200-finetuned-600-FRA-MOS"
BASE_TOKENIZER = "facebook/nllb-200-distilled-600M"
SRC_LANG = "fra_Latn"
TGT_LANG = "mos_Latn"
MAX_NEW_TOKENS = 128


class CT2Translator:
    """Wraps a CTranslate2 Translator to match the translate_batch interface."""

    def __init__(self, model_path: str, device: str = "auto") -> None:
        try:
            import ctranslate2
        except ImportError as e:
            raise ImportError(
                "ctranslate2 is required for CT2 inference. "
                "Install it with: uv sync --extra ctranslate2"
            ) from e
        self._translator = ctranslate2.Translator(model_path, device=device)

    def translate_batch(
        self,
        texts: list[str],
        tokenizer: PreTrainedTokenizerBase,
        src_lang: str,
        tgt_lang: str,
    ) -> list[str]:
        tokenizer.src_lang = src_lang  # type: ignore[attr-defined]
        sources = [
            tokenizer.convert_ids_to_tokens(
                tokenizer(t, add_special_tokens=True)["input_ids"]
            )
            for t in texts
        ]
        results = self._translator.translate_batch(
            sources,
            target_prefix=[[tgt_lang]] * len(texts),
            max_decoding_length=MAX_NEW_TOKENS,
        )
        outputs = []
        for r in results:
            # hypothesis[0] is the lang token — strip it before decoding
            token_ids = tokenizer.convert_tokens_to_ids(r.hypotheses[0][1:])
            outputs.append(tokenizer.decode(token_ids, skip_special_tokens=True))
        return outputs


def _is_ct2_model(path: str) -> bool:
    """True when path is a local directory containing a CTranslate2 model.bin."""
    p = Path(path)
    return p.is_dir() and (p / "model.bin").exists()


def translate(
    text: str,
    model: Seq2SeqModel | CT2Translator,
    tokenizer: PreTrainedTokenizerBase,
    src_lang: str = SRC_LANG,
    tgt_lang: str = TGT_LANG,
) -> str:
    return translate_batch([text], model, tokenizer, src_lang, tgt_lang)[0]


def translate_batch(
    texts: list[str],
    model: Seq2SeqModel | CT2Translator,
    tokenizer: PreTrainedTokenizerBase,
    src_lang: str = SRC_LANG,
    tgt_lang: str = TGT_LANG,
) -> list[str]:
    if isinstance(model, CT2Translator):
        return model.translate_batch(texts, tokenizer, src_lang, tgt_lang)

    inputs = tokenizer(texts, src_lang=src_lang, return_tensors="pt", padding=True, truncation=True)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    outputs = model.generate(
        **inputs,
        forced_bos_token_id=tokenizer.convert_tokens_to_ids(tgt_lang),
        max_new_tokens=MAX_NEW_TOKENS,
        use_cache=True,
    )
    return [str(tokenizer.decode(seq, skip_special_tokens=True)) for seq in outputs]


def load_model(
    model_name: str, tokenizer_name: str = BASE_TOKENIZER
) -> tuple[Seq2SeqModel | CT2Translator, PreTrainedTokenizerBase]:
    tokenizer = cast(
        PreTrainedTokenizerBase,
        AutoTokenizer.from_pretrained(tokenizer_name, src_lang=SRC_LANG, tgt_lang=TGT_LANG),
    )
    if _is_ct2_model(model_name):
        return CT2Translator(model_name), tokenizer

    model = cast(
        Seq2SeqModel,
        AutoModelForSeq2SeqLM.from_pretrained(model_name, device_map="auto"),
    )
    model.eval()
    return model, tokenizer


def main():
    parser = argparse.ArgumentParser(description="French → Mooré translation inference")
    parser.add_argument("text", nargs="?", help="Text to translate (omit for interactive mode)")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Model name or local path")
    parser.add_argument("--src-lang", default=SRC_LANG, help="Source language code")
    parser.add_argument("--tgt-lang", default=TGT_LANG, help="Target language code")
    args = parser.parse_args()

    print(f"Loading model: {args.model}")
    model, tokenizer = load_model(args.model)

    if args.text:
        print(translate(args.text, model, tokenizer, args.src_lang, args.tgt_lang))
    else:
        print("Interactive mode — enter text to translate, Ctrl+C to quit.")
        while True:
            try:
                text = input("> ").strip()
                if text:
                    print(translate(text, model, tokenizer, args.src_lang, args.tgt_lang))
            except (KeyboardInterrupt, EOFError):
                break


if __name__ == "__main__":
    main()
