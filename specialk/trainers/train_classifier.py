"""
CNN Training Runner.

Trains classifier used for style transfer.
Modified version of the following code:
https://github.com/shrimai/Style-Transfer-Through-Back-Translation/blob/master/classifier/cnn_train.py.


TODO: Check this works on both BPE and word-separated tokenized datasets from the huggingface datasets.
"""

from __future__ import division

import argparse
import pickle
import types
from collections.abc import Sequence
from pathlib import Path
from typing import List, Optional, Tuple, Union

import lightning.pytorch as pl
import torch
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.profilers import AdvancedProfiler
from peft import LoraConfig, TaskType
from torch.utils.data import DataLoader, Dataset as TorchDataset

from datasets import Dataset, Value, load_dataset
from specialk.core.constants import LOGGING_DIR, LOGGING_PERF_NAME, PROJECT_DIR
from specialk.core.utils import check_torch_device, log, namespace_to_dict
from specialk.models.classifier.models import (
    BERTClassifier,
    CNNClassifier,
    TextClassifier,
)
from specialk.models.tokenizer import (
    BPEVocabulary,
    HuggingFaceVocabulary,
    SentencePieceVocabulary,
    Vocabulary,
    WordVocabulary,
)
from specialk.trainers.common import _single_split
from specialk.trainers.train import _int_or_float
from specialk.trainers.train_style_decoder import load_seq2seq

DEVICE: str = check_torch_device()
BERT_BASE_NAME = "distilbert/distilbert-base-cased"


def init_dataloader(
    dataset: Dataset,
    tokenizer: Vocabulary,
    batch_size: int,
    shuffle=False,
    n_workers: int = 8,
    persistent_workers: bool = True,
    cache_path: Optional[Union[Path, str]] = None,
) -> DataLoader:
    def tokenize(example):
        # perform tokenization at this stage.
        example["text"] = tokenizer.to_tensor(example["text"])
        return example

    if cache_path:
        if isinstance(cache_path, Path):
            cache_path = str(cache_path)
        if not cache_path.lower().endswith(".parquet"):
            cache_path = f"{cache_path}.parquet"

    tokenized_dataset = dataset.with_format("torch").map(
        tokenize, batched=True, num_proc=n_workers or None, cache_file_name=cache_path
    )
    dataloader = DataLoader(
        tokenized_dataset,
        batch_size=batch_size,
        # DataLoader rejects persistent workers when workers are disabled.
        persistent_workers=persistent_workers and n_workers > 0,
        shuffle=shuffle,
        num_workers=n_workers,
    )

    return dataloader


def get_tokenizer(tokenizer_option: str = "spm", max_length: int = 72) -> Vocabulary:
    # interestingly the bpe tokenizer is a lot slwoer to run instead of the word tokenizer.
    # we should see how well this performs with sentencepience.
    dir_tokenizer = PROJECT_DIR / "assets" / "tokenizer"
    if tokenizer_option == "bpe":
        tokenizer_filepath = dir_tokenizer / "fr_en_bpe"
        tokenizer = BPEVocabulary.from_file(tokenizer_filepath)
        tokenizer.vocab._progress_bar = iter
    elif tokenizer_option == "word":
        # word option.
        tokenizer_filepath = dir_tokenizer / "fr_en_word_moses"
        tokenizer = WordVocabulary.from_file(tokenizer_filepath)
    else:
        # sentencepiece
        tokenizer_filepath = str(dir_tokenizer / "sentencepiece" / "enfr.model")
        tokenizer = SentencePieceVocabulary.from_file(
            tokenizer_filepath, max_length=max_length
        )
    return tokenizer


def get_model(
    model_option: str,
    tokenizer: Vocabulary | None,
    sequence_length: int | None = None,
    learning_rate: float = 0.001,
    name: str = "political",
    vocabulary_size: int | None = None,
    optimizer: str = "adam",
    lr_decay: float = 1.0,
    decay_start: int = 0,
) -> TextClassifier:
    if model_option == "cnn":
        task = CNNClassifier(
            name,
            vocabulary_size=vocabulary_size or tokenizer.vocab_size,
            sequence_length=sequence_length
            or (tokenizer.max_length if tokenizer else None),
            tokenizer=tokenizer,
            learning_rate=learning_rate,
            optimizer=optimizer,
            lr_decay=lr_decay,
            decay_start=decay_start,
        )
    elif model_option == "bert":
        peft_config = LoraConfig(
            task_type=TaskType.SEQ_CLS,
            r=8,
            lora_alpha=32,
            lora_dropout=0.1,
            bias="none",
            target_modules=["q_lin", "k_lin", "v_lin", "out_lin"],
        )

        task = BERTClassifier(
            name=name,
            model_base_name=BERT_BASE_NAME,
            peft_config=peft_config,
        )
    else:
        raise Exception("Model not chosen")

    return task


def _encode_labels(dataset: Dataset) -> Dataset:
    """Turn string labels into integer classes; leave integer labels alone."""
    feature = dataset.features["label"]
    if isinstance(feature, Value) and feature.dtype == "string":
        return dataset.class_encode_column("label")
    return dataset


def _read_labeled_text(path: Path, positive_label: str) -> Dataset:
    texts: list[str] = []
    labels: list[int] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        label, text = line.split(" ", 1)
        texts.append(text)
        labels.append(1 if label == positive_label else 0)
    return Dataset.from_dict({"text": texts, "label": labels})


def load_text_classification_dataset(
    train_path: Path, val_path: Path, positive_label: str
) -> tuple[Dataset, Dataset]:
    """Load the old ``label text`` files used by the 2019 classifiers."""
    return (
        _read_labeled_text(train_path, positive_label),
        _read_labeled_text(val_path, positive_label),
    )


class _LegacyDictUnpickler(pickle.Unpickler):
    def find_class(self, module: str, name: str):
        if module.startswith("onmt") and name == "Dict":
            from specialk.models.tokenizers.dict import Dict

            return Dict
        return super().find_class(module, name)


def load_legacy_tokenized_data(path: Path) -> dict:
    """Load the old ``political.train.pt`` file with its ``onmt.Dict``."""
    module = types.ModuleType("legacy_tokenized")
    module.Unpickler = _LegacyDictUnpickler
    module.load = pickle.load
    return torch.load(
        path, map_location="cpu", weights_only=False, pickle_module=module
    )


class _TokenizedDataset(TorchDataset):
    def __init__(self, rows: list[torch.Tensor], labels: list[torch.Tensor]):
        self.rows = rows
        self.labels = labels

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        return {"text": self.rows[index], "label": self.labels[index]}


def load_tokenized_classification_dataset(
    path: Path,
    batch_size: int,
    positive_label: str,
    n_workers: int = 0,
) -> tuple[DataLoader, DataLoader, int, int]:
    """Load the old pre-tokenized political classifier data.

    The labels are republican=1, so a democratic-positive run flips them.
    """
    data = load_legacy_tokenized_data(path)

    def rows(split):
        source = [row.long() for row in data[split]["src"]]
        labels = [label.long().squeeze(0) for label in data[split]["tgt"]]
        if positive_label == "democratic":
            labels = [1 - label for label in labels]
        return source, labels

    train_rows, train_labels = rows("train")
    val_rows, val_labels = rows("valid")
    vocabulary_size = data["dicts"]["src"].size()
    sequence_length = train_rows[0].size(0)

    train_loader = DataLoader(
        _TokenizedDataset(train_rows, train_labels),
        batch_size=batch_size,
        shuffle=True,
        num_workers=n_workers,
    )
    val_loader = DataLoader(
        _TokenizedDataset(val_rows, val_labels),
        batch_size=batch_size,
        shuffle=False,
        num_workers=n_workers,
    )
    return train_loader, val_loader, sequence_length, vocabulary_size


def init_logger(args: argparse.Namespace, project_name: str, params: dict):
    """Log to W&B by default, or to TensorBoard with ``--logger tensorboard``."""
    if args.logger == "wandb":
        from lightning.pytorch.loggers import WandbLogger

        name = args.run_name
        if name is None and not args.wandb_run_id:
            name = project_name.replace("/", "-")
        extra = {}
        if args.wandb_run_id:
            extra = {"id": args.wandb_run_id, "resume": "allow"}
        return WandbLogger(
            project=args.wandb_project,
            name=name,
            save_dir=PROJECT_DIR,
            config=params,
            **extra,
        )
    logger = TensorBoardLogger(LOGGING_DIR, name=project_name)
    logger.log_hyperparams(params=params)
    return logger


def main_new(args):
    batch_size = args.batch_size

    # tokenizer
    vocabulary_size = None
    if args.tokenized_path:
        (
            train_dataloader,
            val_dataloader,
            sequence_length,
            vocabulary_size,
        ) = load_tokenized_classification_dataset(
            args.tokenized_path, batch_size, args.positive_label, args.n_workers
        )
        tokenizer = None
        dataset_name = f"local/{args.tokenized_path.stem}"
    else:
        if args.model_type == "bert":
            tokenizer = HuggingFaceVocabulary(
                name=BERT_BASE_NAME,
                pretrained_model_name_or_path=BERT_BASE_NAME,
                max_length=args.max_seq_len or 512,
            )
            sequence_length = tokenizer.max_length
        elif args.tokenizer_path:
            tokenizer = WordVocabulary.from_file(args.tokenizer_path)
            sequence_length = tokenizer.max_length
        elif args.mt_checkpoint:
            mt_model = load_seq2seq(args.mt_checkpoint)
            tokenizer, sequence_length = mt_model.tokenizer, mt_model.sequence_length
        else:
            tokenizer = get_tokenizer(args.tokenizer, args.max_seq_len or 72)
            sequence_length = tokenizer.max_length

    task = get_model(
        args.model_type,
        tokenizer,
        sequence_length=sequence_length,
        learning_rate=args.lr,
        name=args.model_name,
        vocabulary_size=vocabulary_size,
        optimizer=args.optimizer,
        lr_decay=args.lr_decay,
        decay_start=args.decay_start,
    )

    log.info("Loaded tokenizer", tokenizer=tokenizer)

    if not args.tokenized_path:
        if args.train_path:
            if not args.val_path:
                raise ValueError("--val-path is required with --train-path")
            train_dataset, val_dataset = load_text_classification_dataset(
                args.train_path, args.val_path, args.positive_label
            )
            dataset_name = f"local/{args.train_path.stem}"
        else:
            # publications uses the default config, political the ``classifier`` one.
            dataset = load_dataset(args.dataset_name, args.config)
            train_dataset = _encode_labels(_single_split(dataset, "train"))
            val_dataset = _encode_labels(
                _single_split(dataset, "eval", "validation", "valid", "test")
            )
            dataset_name = args.dataset_name

        train_dataloader = init_dataloader(
            train_dataset,
            tokenizer,
            batch_size,
            shuffle=True,
            n_workers=args.n_workers,
        )
        val_dataloader = init_dataloader(
            val_dataset,
            tokenizer,
            batch_size,
            shuffle=False,
            n_workers=args.n_workers,
        )

    project_name = f"{dataset_name}/{task.name}"
    checkpoint_dir = (
        Path(args.checkpoint_dir)
        if args.checkpoint_dir
        else Path.cwd() / "checkpoints"
    )
    resume_from = None
    if args.resume:
        resume_from = (
            checkpoint_dir / "last.ckpt"
            if args.resume == "last"
            else Path(args.resume)
        )
        if not resume_from.exists():
            raise SystemExit(f"no checkpoint at {resume_from}")

    params = {
        "batch_size": batch_size,
        "tokenizer": tokenizer.__class__.__name__ if tokenizer else "legacy",
        "dataset": dataset_name,
        "tokenizer_path": str(args.tokenizer_path or args.mt_checkpoint)
        if args.tokenizer_path or args.mt_checkpoint
        else None,
        "tokenized_path": str(args.tokenized_path) if args.tokenized_path else None,
        "optimizer": args.optimizer,
        "learning_rate": args.lr,
        "lr_decay": args.lr_decay,
        "decay_start": args.decay_start,
        "max_epochs": args.epochs,
        "resume_from": str(resume_from) if resume_from else None,
    }
    logger = init_logger(args, project_name, params)

    profiler = AdvancedProfiler(
        dirpath=logger.log_dir or LOGGING_DIR, filename=LOGGING_PERF_NAME
    )

    # 16-bit mixed precision is a CUDA recipe; MPS runs in full precision.
    resolved_accelerator = DEVICE if args.accelerator == "auto" else args.accelerator
    precision = "16-mixed" if resolved_accelerator == "cuda" else "32-true"

    trainer = pl.Trainer(
        accelerator=args.accelerator,
        max_epochs=args.epochs,
        log_every_n_steps=20,
        logger=logger,
        profiler=profiler,
        precision=precision,
        limit_train_batches=args.limit_train_batches,
        limit_val_batches=args.limit_val_batches,
        val_check_interval=1.0,
        gradient_clip_val=args.gradient_clip_val,
        gradient_clip_algorithm="norm",
        callbacks=[
            ModelCheckpoint(dirpath=checkpoint_dir, save_last=True, save_top_k=-1)
        ],
    )

    trainer.fit(
        task,
        train_dataloaders=train_dataloader,
        val_dataloaders=val_dataloader,
        ckpt_path=resume_from,
    )
    return trainer


def get_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="CNN Training Runner")
    parser.add_argument(
        "--model_type",
        choices=["cnn", "bert"],
        default="cnn",
        help="Type of model to use",
    )
    parser.add_argument(
        "--model_name", type=str, default="political", help="Name of the model"
    )
    parser.add_argument(
        "--tokenizer",
        choices=["bpe", "word", "spm"],
        default="spm",
        help="Type of tokenizer to use (for CNN model)",
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        default="thien/political",
        help="Name of the dataset to use",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Dataset config; political uses 'classifier', publications the default",
    )
    parser.add_argument(
        "--mt-checkpoint",
        type=Path,
        default=None,
        help="seq2seq checkpoint whose tokenizer and sequence length the CNN shares",
    )
    parser.add_argument(
        "--tokenizer-path",
        type=Path,
        default=None,
        help="WordVocabulary JSON to load instead of building a tokenizer",
    )
    parser.add_argument(
        "--tokenized-path",
        type=Path,
        default=None,
        help="old pre-tokenized .train.pt file with an onmt.Dict",
    )
    parser.add_argument(
        "--train-path",
        type=Path,
        default=None,
        help="labelled text file with 'label text' lines",
    )
    parser.add_argument(
        "--val-path",
        type=Path,
        default=None,
        help="validation file for --train-path",
    )
    parser.add_argument(
        "--positive-label",
        choices=["democratic", "republican"],
        default="democratic",
        help="label that the sigmoid output should predict",
    )
    parser.add_argument(
        "--optimizer",
        choices=["adam", "sgd"],
        default="adam",
        help="optimizer for the CNN",
    )
    parser.add_argument(
        "--lr-decay",
        type=float,
        default=1.0,
        help="learning-rate decay factor after --decay-start",
    )
    parser.add_argument(
        "--decay-start",
        type=int,
        default=0,
        help="epoch after which the learning rate starts decaying",
    )
    parser.add_argument(
        "--gradient-clip-val",
        type=float,
        default=0.5,
        help="gradient clipping value",
    )
    parser.add_argument(
        "--batch_size", type=int, default=128, help="Batch size for training"
    )
    parser.add_argument(
        "--max-seq-len",
        type=int,
        default=None,
        help="Maximum sequence length for the tokenizer (spm defaults to 72)",
    )
    parser.add_argument(
        "--epochs", type=int, default=2, help="Number of epochs for training"
    )
    parser.add_argument(
        "--lr", type=float, default=0.001, help="Learning rate for the CNN"
    )
    parser.add_argument(
        "--accelerator", choices=["auto", "cpu", "mps", "cuda"], default="auto"
    )
    parser.add_argument(
        "--limit-train-batches",
        type=_int_or_float,
        default=1.0,
        help="batches (int, e.g. 2) or fraction of an epoch (float, e.g. 0.5)",
    )
    parser.add_argument(
        "--limit-val-batches",
        type=_int_or_float,
        default=1.0,
        help="batches (int) or fraction of the validation set (float)",
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        default=None,
        help="where to write checkpoints (default: ./checkpoints)",
    )
    parser.add_argument(
        "--resume",
        nargs="?",
        const="last",
        default=None,
        help="resume from a checkpoint, or from the run's last.ckpt when no path is given",
    )
    parser.add_argument(
        "--logger",
        choices=["wandb", "tensorboard"],
        default="wandb",
        help="experiment logger",
    )
    parser.add_argument(
        "--wandb-project",
        default="specialk",
        help="W&B project to log to",
    )
    parser.add_argument(
        "--run-name",
        default=None,
        help="logger run name (default: dataset and model name)",
    )
    parser.add_argument(
        "--wandb-run-id",
        default=None,
        help="continue an existing W&B run when resuming",
    )
    parser.add_argument(
        "--n-workers",
        type=int,
        default=0,
        help="dataloader workers",
    )

    args = parser.parse_args(argv)
    return args


if __name__ == "__main__":
    main_new(get_args())
