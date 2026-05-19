from dataclasses import dataclass, field
from typing import cast

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
    TrainerCallback,
    Seq2SeqTrainingArguments,
)

load_dotenv()


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
    eval_subset_size: int = field(
        default=500,
        metadata={"help": "Number of validation examples used for BLEU/chrF++ during training"},
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


class TestEvaluationCallback(TrainerCallback):
    def __init__(self, test_dataset):
        self.test_dataset = test_dataset
        self.has_run = False

    def on_step_end(self, args, state, control, **kwargs):
        # Run only once, at the very end of training
        if self.has_run:
            return control

        if state.global_step >= state.max_steps:
            trainer = kwargs["trainer"]

            test_res = trainer.evaluate(self.test_dataset, metric_key_prefix="test")
            trainer.log(test_res)
            trainer.save_metrics("test", test_res)

            self.has_run = True

        return control


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
        range(min(data_args.eval_subset_size, len(tokenized_dataset["validation"])))
    )

    data_collator = DataCollatorForSeq2Seq(
        tokenizer, model=model, padding=True, pad_to_multiple_of=8
    )
    test_dataset = None
    if "test" in tokenized_dataset:
        if data_args.max_train_samples > 0:
            test_dataset = tokenized_dataset["test"].select(
                range(min(data_args.eval_subset_size, len(tokenized_dataset["test"])))
            )
        else:
            test_dataset = tokenized_dataset["test"]
    test_eval_callback = TestEvaluationCallback(test_dataset)

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=eval_dataset,
        compute_metrics=build_compute_metrics(tokenizer),
        callbacks=[
            EarlyStoppingCallback(early_stopping_patience=model_args.early_stopping_patience),
            test_eval_callback,
        ],
        processing_class=tokenizer,
        data_collator=data_collator,
    )

    trainer.train(resume_from_checkpoint=training_args.resume_from_checkpoint)

    # eval_res = trainer.evaluate(tokenized_dataset["validation"], metric_key_prefix="eval_final")
    # trainer.save_metrics("eval", eval_res)
    trainer.push_to_hub()


if __name__ == "__main__":
    main()
