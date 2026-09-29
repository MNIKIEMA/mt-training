import logging
import os
from contextlib import nullcontext
from dataclasses import dataclass, field
from importlib.util import find_spec
from pathlib import Path
from typing import cast
from unittest.mock import patch

import evaluate
import numpy as np
import torch
from datasets import load_dataset
from dotenv import load_dotenv
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    EarlyStoppingCallback,
    HfArgumentParser,
    PreTrainedTokenizerBase,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
)

from mt_training.eval import (
    FLORES_DEFAULT_SPLIT,
    FLORES_PLUS,
    EvalConfig,
    run_evaluation,
)
from mt_training.text import normalize_for_nllb

load_dotenv()
logger = logging.getLogger(__name__)


@dataclass
class DataTrainingArguments:
    dataset_id: str = field(
        default="madoss/moore-web-parallel",
        metadata={"help": "HuggingFace dataset ID"},
    )
    dataset_config: str | None = field(
        default=None,
        metadata={"help": "Dataset config name (e.g. mos-fra); None for single-config datasets"},
    )
    dataset_revision: str | None = field(
        default=None,
        metadata={"help": "Dataset tag or commit to pin (e.g. v1.0.0); None = latest"},
    )
    source_field: str = field(
        default="french",
        metadata={"help": "Dataset column translated from (use `moore` for Mooré → French)"},
    )
    target_field: str = field(
        default="moore",
        metadata={"help": "Dataset column translated into (use `french` for Mooré → French)"},
    )
    src_lang: str = field(
        default="fra_Latn",
        metadata={"help": "Source language code (NLLB format, e.g. fra_Latn)"},
    )
    tgt_lang: str = field(
        default="mos_Latn",
        metadata={"help": "Target language code (NLLB format, e.g. mos_Latn)"},
    )
    max_length: int = field(
        default=256,
        metadata={"help": "Max token length for source and target sequences"},
    )
    validation_size: int = field(
        default=100,
        metadata={"help": "Number of validation examples used for BLEU/chrF++ during training"},
    )
    post_training_eval_limit: int = field(
        default=-1,
        metadata={
            "help": (
                "Limit each post-training CT2 eval to this many examples "
                "(-1 = evaluate the full dataset)"
            )
        },
    )
    max_train_samples: int = field(
        default=-1,
        metadata={
            "help": "Truncate training set to this many examples before tokenization (-1 = use all)"
        },
    )


@dataclass
class ModelArguments:
    model_name: str = field(
        default="facebook/nllb-200-distilled-600M",
        metadata={"help": "Pretrained model name or path"},
    )
    hf_id: str = field(
        default="madoss",
        metadata={"help": "HuggingFace user/org ID for output repo"},
    )
    early_stopping_patience: int = field(
        default=3,
        metadata={"help": "Stop training after this many evals with no improvement"},
    )
    output_dir_root: str = field(
        default=".",
        metadata={"help": "Root directory under which repo_name subdirectory is created"},
    )
    repo_name: str = field(
        default="",
        metadata={"help": "HuggingFace repo name; defaults to nllb-200-finetuned-600-{SRC}-{TGT}"},
    )
    ct2_quantization: str = field(
        default="int8",
        metadata={"help": "CTranslate2 quantization for post-training evaluation"},
    )
    model_dtype: str | None = field(
        default=None,
        metadata={
            "help": (
                "Load the model weights in this dtype (e.g. bfloat16) to fit small GPUs; "
                "None = float32. Pure bf16 weights are for smoke tests, not real runs."
            )
        },
    )


def map_columns(dataset, data_args: DataTrainingArguments):
    """Rename to the training contract: `source` -> `data_source`, then the chosen
    source/target text columns -> `source`/`target` (e.g. french/moore, or moore/french)."""
    if data_args.source_field == data_args.target_field:
        raise ValueError("source_field and target_field must differ")
    dataset = dataset.rename_column("source", "data_source")
    dataset = dataset.rename_column(data_args.source_field, "source")
    return dataset.rename_column(data_args.target_field, "target")


def load_and_prepare_dataset(data_args: DataTrainingArguments):
    dataset = load_dataset(
        data_args.dataset_id,
        name=data_args.dataset_config,
        revision=data_args.dataset_revision,
    )
    return map_columns(dataset, data_args)


def build_tokenize_fn(tokenizer: PreTrainedTokenizerBase, data_args: DataTrainingArguments):
    # The language tokens come from the tokenizer's src_lang/tgt_lang, set in main().
    # Both sides are normalized for the NLLB vocabulary (’ « » -> ' "), see text.py.
    # The NLLB tokenizer silently ignores src_lang/tgt_lang passed to __call__.
    def tokenize_fn(examples):
        tokenized = tokenizer(
            [normalize_for_nllb(t) for t in examples["source"]],
            text_target=[normalize_for_nllb(t) for t in examples["target"]],
            max_length=data_args.max_length,
            truncation=True,
        )
        return {
            "input_ids": tokenized["input_ids"],
            "attention_mask": tokenized["attention_mask"],
            "labels": tokenized["labels"],
        }

    return tokenize_fn


def build_compute_metrics(tokenizer):
    bleu_metric = evaluate.load("sacrebleu")
    chrf_metric = evaluate.load("chrf")

    def compute_metrics(eval_preds):
        preds, labels = eval_preds
        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
        decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
        decoded_labels = [
            [label] for label in tokenizer.batch_decode(labels, skip_special_tokens=True)
        ]
        bleu_result = (
            bleu_metric.compute(predictions=decoded_preds, references=decoded_labels) or {}
        )  # type: ignore[union-attr]
        chrf_result = (
            chrf_metric.compute(predictions=decoded_preds, references=decoded_labels, word_order=2)
            or {}
        )  # type: ignore[union-attr]
        return {
            "bleu": bleu_result.get("score", 0.0),
            "chrf++": chrf_result.get("score", 0.0),
        }

    return compute_metrics


def wandb_finish_context(report_to):
    if isinstance(report_to, str):
        report_to = [report_to]

    if report_to and ("wandb" in report_to or "all" in report_to) and find_spec("wandb"):
        return patch("wandb.finish")

    return nullcontext()


def convert_to_ct2(model_dir: str, output_dir: str, quantization: str) -> str:
    import ctranslate2

    converter = ctranslate2.converters.TransformersConverter(
        model_dir,
        low_cpu_mem_usage=True,
    )
    return str(converter.convert(output_dir, quantization=quantization, force=True))


def log_metrics_to_wandb(metrics: dict[str, float], prefix: str) -> None:
    if not find_spec("wandb"):
        return

    import wandb

    if wandb.run is None:
        return

    wandb.log({f"{prefix}/{key}": value for key, value in metrics.items()})


def strip_metric_prefix(metrics: dict[str, float], prefix: str) -> dict[str, float]:
    metric_prefix = f"{prefix}_"
    return {key.removeprefix(metric_prefix): value for key, value in metrics.items()}


def evaluate_test_split(trainer, tokenized_dataset, data_args) -> None:
    if "test" not in tokenized_dataset:
        logger.info("Skipping test split eval: no test split found")
        return

    test_dataset = tokenized_dataset["test"]
    if data_args.post_training_eval_limit > 0:
        test_dataset = test_dataset.select(
            range(min(data_args.post_training_eval_limit, len(test_dataset)))
        )

    logger.info(
        "Running test split eval on %s samples (limit=%s)",
        len(test_dataset),
        data_args.post_training_eval_limit if data_args.post_training_eval_limit > 0 else "all",
    )
    metrics = trainer.evaluate(test_dataset, metric_key_prefix="test")
    trainer.save_metrics("test", metrics)
    log_metrics_to_wandb(strip_metric_prefix(metrics, "test"), "test")


def finish_wandb_run() -> None:
    if not find_spec("wandb"):
        return

    import wandb

    if wandb.run is not None:
        wandb.finish()


def run_post_training_ct2_evaluations(model_args, data_args, training_args, trainer) -> None:
    trainer.save_model(training_args.output_dir)
    ct2_output_dir = f"{Path(training_args.output_dir)}-ct2"
    ct2_model = convert_to_ct2(
        training_args.output_dir,
        ct2_output_dir,
        model_args.ct2_quantization,
    )

    eval_limit = (
        data_args.post_training_eval_limit if data_args.post_training_eval_limit > 0 else None
    )

    evals = [
        (
            "flores_plus",
            EvalConfig(
                model=ct2_model,
                dataset=FLORES_PLUS,
                src_lang=data_args.src_lang,
                tgt_lang=data_args.tgt_lang,
                split=FLORES_DEFAULT_SPLIT,
                limit=eval_limit,
            ),
        ),
    ]

    saved_metrics: dict[str, float] = {}
    for prefix, cfg in evals:
        logger.info(
            "Running post-training CT2 eval %s on %s (limit=%s)",
            prefix,
            cfg.dataset,
            cfg.limit if cfg.limit is not None else "all",
        )
        metrics, _, _, _ = run_evaluation(cfg)
        log_metrics_to_wandb(metrics, prefix)
        saved_metrics.update({f"{prefix}_{key}": value for key, value in metrics.items()})

    trainer.save_metrics("post_training_ct2_eval", saved_metrics)


def main():
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, Seq2SeqTrainingArguments))  # ty:ignore[invalid-argument-type]
    model_args, data_args, training_args = cast(
        tuple[ModelArguments, DataTrainingArguments, Seq2SeqTrainingArguments],
        parser.parse_args_into_dataclasses(),
    )
    src_tag = data_args.src_lang.split("_")[0].upper()
    tgt_tag = data_args.tgt_lang.split("_")[0].upper()
    repo_name = model_args.repo_name or f"nllb-200-finetuned-600-{src_tag}-{tgt_tag}"
    training_args.output_dir = f"{model_args.output_dir_root}/{repo_name}"
    training_args.hub_model_id = f"{model_args.hf_id}/{repo_name}"
    training_args.run_name = training_args.run_name or repo_name
    training_args.load_best_model_at_end = True
    training_args.metric_for_best_model = "chrf++"
    training_args.greater_is_better = True
    os.environ.setdefault("WANDB_PROJECT", training_args.project)

    dataset = load_and_prepare_dataset(data_args)
    if data_args.max_train_samples > 0:
        dataset["train"] = dataset["train"].select(range(data_args.max_train_samples))

    tokenizer = cast(
        PreTrainedTokenizerBase,
        AutoTokenizer.from_pretrained(
            model_args.model_name,
            src_lang=data_args.src_lang,
            tgt_lang=data_args.tgt_lang,
        ),
    )
    model = AutoModelForSeq2SeqLM.from_pretrained(
        model_args.model_name,
        device_map="auto",
        use_cache=False,
        dtype=getattr(torch, model_args.model_dtype) if model_args.model_dtype else None,
    )

    tokenized_dataset = dataset.map(
        build_tokenize_fn(tokenizer, data_args),
        batched=True,
        remove_columns=dataset["train"].column_names,
        desc="Tokenizing dataset",
    )

    torch.cuda.empty_cache()

    # Shuffle before taking the subset: splits can be stored grouped by source.
    eval_dataset = (
        tokenized_dataset["validation"]
        .shuffle(seed=training_args.seed)
        .select(range(min(data_args.validation_size, len(tokenized_dataset["validation"]))))
    )

    data_collator = DataCollatorForSeq2Seq(
        tokenizer, model=model, padding=True, pad_to_multiple_of=8
    )
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=eval_dataset,
        compute_metrics=build_compute_metrics(tokenizer),
        callbacks=[
            EarlyStoppingCallback(early_stopping_patience=model_args.early_stopping_patience),
        ],
        processing_class=tokenizer,
        data_collator=data_collator,
    )

    with wandb_finish_context(training_args.report_to):
        trainer.train(resume_from_checkpoint=training_args.resume_from_checkpoint)

    try:
        evaluate_test_split(trainer, tokenized_dataset, data_args)
        run_post_training_ct2_evaluations(model_args, data_args, training_args, trainer)
        if training_args.push_to_hub:
            trainer.push_to_hub()
    finally:
        finish_wandb_run()


if __name__ == "__main__":
    main()
