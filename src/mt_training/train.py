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
    DEFAULT_DATASET,
    FLORES_DEFAULT_SPLIT,
    FLORES_PLUS,
    EvalConfig,
    run_evaluation,
)

load_dotenv()
logger = logging.getLogger(__name__)


@dataclass
class DataTrainingArguments:
    dataset_id: str = field(
        default="madoss/fr-mos-final-data",
        metadata={"help": "HuggingFace dataset ID"},
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


def load_and_prepare_dataset(data_args: DataTrainingArguments):
    dataset = load_dataset(data_args.dataset_id)
    dataset = dataset.rename_column("source", "data_source")
    dataset = dataset.rename_column("french", "source")
    dataset = dataset.rename_column("moore", "target")

    def add_language_info(example):
        example["source_lang"] = data_args.src_lang
        example["target_lang"] = data_args.tgt_lang
        return example

    return dataset.map(add_language_info)


def build_tokenize_fn(tokenizer: PreTrainedTokenizerBase, data_args: DataTrainingArguments):
    def tokenize_fn(examples):
        input_ids_list = []
        attention_mask_list = []
        labels_list = []

        for src, tgt, src_lang, tgt_lang in zip(
            examples["source"],
            examples["target"],
            examples["source_lang"],
            examples["target_lang"],
            strict=True,
        ):
            tokenized = tokenizer(
                src,
                text_target=tgt,
                src_lang=src_lang,
                tgt_lang=tgt_lang,
                max_length=data_args.max_length,
                truncation=True,
            )
            input_ids_list.append(tokenized["input_ids"])
            attention_mask_list.append(tokenized["attention_mask"])
            labels_list.append(tokenized["labels"])

        return {
            "input_ids": input_ids_list,
            "attention_mask": attention_mask_list,
            "labels": labels_list,
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
        data_args.post_training_eval_limit
        if data_args.post_training_eval_limit > 0
        else "all",
    )
    metrics = trainer.evaluate(test_dataset, metric_key_prefix="test")
    trainer.save_metrics("test", metrics)
    log_metrics_to_wandb(metrics, "test")


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
            "s3_eval",
            EvalConfig(
                model=ct2_model,
                dataset=DEFAULT_DATASET,
                src_lang=data_args.src_lang,
                tgt_lang=data_args.tgt_lang,
                limit=eval_limit,
            ),
        ),
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
        model_args.model_name, device_map="auto", use_cache=False
    )

    tokenized_dataset = dataset.map(
        build_tokenize_fn(tokenizer, data_args),
        batched=True,
        remove_columns=dataset["train"].column_names,
        desc="Tokenizing dataset",
    )

    torch.cuda.empty_cache()

    eval_dataset = tokenized_dataset["validation"].select(
        range(min(data_args.validation_size, len(tokenized_dataset["validation"])))
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
        trainer.push_to_hub()
    finally:
        finish_wandb_run()


if __name__ == "__main__":
    main()
