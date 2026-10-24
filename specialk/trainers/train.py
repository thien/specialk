"""Unified training entry point for the machine-translation models.

Examples:
    # one training batch on the small local corpus (CPU or Apple Silicon)
    python -m specialk.trainers.train --dataset dummy \\
        --max-epochs 1 --limit-train-batches 1 --limit-val-batches 1

    # production-size model on the full EN-FR corpus
    python -m specialk.trainers.train --dataset mt --prod --accelerator cuda

    # French -> English publications decoder
    python -m specialk.trainers.train --dataset publications --model transformer
"""

from __future__ import annotations, division

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Sequence, Union

import lightning.pytorch as pl
import pandas as pd
import torch
from datasets import Dataset
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.profilers import AdvancedProfiler

from specialk.core.constants import (
    LOGGING_DIR,
    LOGGING_PERF_NAME,
    RNN,
    SEED,
    TRANSFORMER,
)
from specialk.core.utils import log
from specialk.models.mt_model import RNNModule, TransformerModule
from specialk.models.transformer.torch.pytorch_transformer import (
    PyTorchTransformerModule,
)
from specialk.trainers.common import (
    CACHE_DIR,
    DATASET_DIR,
    TOKENIZER_OPTIONS,
    cache_filename,
    init_dataloader,
    invert_df_columns,
    load_mono_tokenizer,
    load_mt_training_dataset,
    load_mt_validation_dataset,
    load_publications_datasets,
    load_tokenizer,
)

DATASETS = ("dummy", "mt", "publications")


@dataclass(frozen=True)
class DatasetBundle:
    """The train/validation datasets plus the names used for logging."""

    task_name: str
    train: Dataset
    valid: Dataset
    train_path: Union[str, Path]
    valid_path: Union[str, Path]


def load_datasets(
    dataset: str = "dummy",
    train_path: Optional[Union[str, Path]] = None,
    valid_path: Optional[Union[str, Path]] = None,
    back_translation: bool = False,
    n_workers: int = 8,
) -> DatasetBundle:
    """Load the requested dataset and normalise its columns to source/target."""
    if dataset == "dummy":
        train_path = train_path or DATASET_DIR / "de_en_30k_train.parquet"
        valid_path = valid_path or DATASET_DIR / "de_en_30k_valid.parquet"
        log.info("Loading datasets", train=str(train_path), valid=str(valid_path))

        df_train = pd.read_parquet(train_path).sample(frac=1, random_state=SEED)
        df_valid = pd.read_parquet(valid_path)
        if back_translation:
            log.info("Backtranslation flag enabled; inverting dataset columns.")
            df_train = invert_df_columns(df_train)
            df_valid = invert_df_columns(df_valid)
        else:
            log.info("Backtranslation is disabled.")

        return DatasetBundle(
            task_name="nmt_model_dummy",
            train=Dataset.from_pandas(df_train),
            valid=Dataset.from_pandas(df_valid),
            train_path=train_path,
            valid_path=valid_path,
        )

    if dataset == "mt":
        src_lang, tgt_lang = ("en", "fr") if back_translation else ("fr", "en")
        log.info(
            "Loading machine-translation datasets",
            src_lang=src_lang,
            tgt_lang=tgt_lang,
        )
        return DatasetBundle(
            task_name="nmt_model",
            train=load_mt_training_dataset(src_lang, tgt_lang, num_proc=n_workers),
            valid=load_mt_validation_dataset(src_lang, tgt_lang, num_proc=n_workers),
            train_path="thien_mt_datasets",
            valid_path="wmt15",
        )

    if dataset == "publications":
        if back_translation:
            raise ValueError(
                "Backtranslation is not supported for the publications dataset."
            )
        train, valid = load_publications_datasets(num_proc=n_workers)
        return DatasetBundle(
            task_name="decoder_model",
            train=train,
            valid=valid,
            train_path="thien_publications_train",
            valid_path="thien_publications_eval",
        )

    raise ValueError(f"Unknown dataset {dataset!r}; expected one of {DATASETS}")


def resolve_hyperparameters(
    model: str = TRANSFORMER,
    prod: bool = False,
    batch_size: Optional[int] = None,
    max_seq_len: Optional[int] = None,
    n_epochs: Optional[int] = None,
    learning_rate: float = 0.001,
) -> Dict[str, Any]:
    """Return the batch/model configuration for a production or smoke run.

    ``prod`` selects the full-size model, independently of which device is
    available, so MPS machines can also train on the real corpus.
    """
    if model not in {RNN, TRANSFORMER}:
        raise ValueError(
            f"Unknown model {model!r}; expected one of {RNN!r}, {TRANSFORMER!r}"
        )

    if prod:
        defaults = {
            "batch_size": 192 if model == RNN else 64,
            "max_seq_len": 75 if model == RNN else 100,
            "n_epochs": 6,
        }
    else:
        # A short sanity-check run: small model, small sequences.
        defaults = {"batch_size": 192, "max_seq_len": 30, "n_epochs": 60}

    if model == RNN:
        model_config = {
            "name": "lstm" if prod else "lstm_smol",
            "brnn": True,
            "learning_rate": learning_rate,
            "accumulate_grad_batches": 1,
        }
        if not prod:
            model_config["rnn_size"] = 192
            model_config["d_word_vec"] = 128
    else:
        model_config = {
            "name": "transformer" if prod else "transformer_smol",
            "num_encoder_layers": 6 if prod else 3,
            "num_decoder_layers": 6 if prod else 3,
            "n_heads": 8 if prod else 4,
            "dim_model": 512 if prod else 128,
            "n_warmup_steps": 4000 if prod else 80,
            "learning_rate": learning_rate,
            # multiply by batch_size to get effective batch size
            "accumulate_grad_batches": 10 if prod else 4,
        }

    return {
        "batch_size": defaults["batch_size"] if batch_size is None else batch_size,
        "max_seq_len": defaults["max_seq_len"] if max_seq_len is None else max_seq_len,
        "n_epochs": defaults["n_epochs"] if n_epochs is None else n_epochs,
        "model_config": model_config,
    }


def train(
    model: str = TRANSFORMER,
    dataset: str = "dummy",
    tokenizer: str = "spm",
    prod: bool = False,
    legacy_transformer: bool = False,
    back_translation: bool = False,
    batch_size: Optional[int] = None,
    max_seq_len: Optional[int] = None,
    n_epochs: Optional[int] = None,
    learning_rate: float = 0.001,
    precision: str = "16-mixed",
    accelerator: str = "auto",
    devices: Union[str, int] = "auto",
    limit_train_batches: Union[int, float] = 1.0,
    limit_val_batches: Union[int, float] = 1.0,
    val_check_interval: Union[int, float] = 1.0,
    log_every_n_steps: int = 20,
    n_workers: int = 7,
    seed: int = SEED,
    train_path: Optional[Union[str, Path]] = None,
    valid_path: Optional[Union[str, Path]] = None,
) -> pl.Trainer:
    """Train a model and return the Lightning trainer."""
    pl.seed_everything(seed, workers=True)
    torch.use_deterministic_algorithms(True)

    hyperparameters = resolve_hyperparameters(
        model=model,
        prod=prod,
        batch_size=batch_size,
        max_seq_len=max_seq_len,
        n_epochs=n_epochs,
        learning_rate=learning_rate,
    )

    bundle = load_datasets(
        dataset=dataset,
        train_path=train_path,
        valid_path=valid_path,
        back_translation=back_translation,
        n_workers=n_workers,
    )

    if dataset == "dummy":
        src_tokenizer, tgt_tokenizer, src_tokenizer_path, tgt_tokenizer_path = (
            load_mono_tokenizer()
        )
    else:
        src_tokenizer, src_tokenizer_path = load_tokenizer(
            tokenizer, hyperparameters["max_seq_len"]
        )
        tgt_tokenizer, tgt_tokenizer_path = src_tokenizer, src_tokenizer_path

    tokenizer_config = {
        "tokenizer": src_tokenizer,
        "decoder_tokenizer": tgt_tokenizer,
        "vocabulary_size": src_tokenizer.vocab_size,
        "decoder_vocabulary_size": tgt_tokenizer.vocab_size,
        "sequence_length": src_tokenizer.max_length,
    }
    model_config = {**hyperparameters["model_config"], **tokenizer_config}
    if legacy_transformer and not prod:
        model_config["name"] += "_legacy"

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_settings = {
        "tokenizer": tokenizer,
        "max_seq_len": hyperparameters["max_seq_len"],
        "back_translation": back_translation,
    }
    train_dataloader, _ = init_dataloader(
        bundle.train,
        src_tokenizer,
        hyperparameters["batch_size"],
        decoder_tokenizer=tgt_tokenizer,
        shuffle=True,
        cache_path=CACHE_DIR / cache_filename(bundle.train_path, **cache_settings),
        n_workers=n_workers,
    )
    val_dataloader, _ = init_dataloader(
        bundle.valid,
        src_tokenizer,
        hyperparameters["batch_size"],
        decoder_tokenizer=tgt_tokenizer,
        shuffle=False,
        cache_path=CACHE_DIR / cache_filename(bundle.valid_path, **cache_settings),
        n_workers=n_workers,
    )
    log.info("Created dataset dataloaders.")

    if model == RNN:
        task = RNNModule(**model_config)
        # is this needed?
        task.model.PAD = src_tokenizer.PAD
    elif legacy_transformer and not prod:
        task = TransformerModule(**model_config)
    else:
        task = PyTorchTransformerModule(**model_config)

    log.info("model initialised.", model=task)

    logging_hyperparameters = {
        "batch_size": hyperparameters["batch_size"],
        "src_tokenizer": src_tokenizer.__class__.__name__,
        "tgt_tokenizer": tgt_tokenizer.__class__.__name__,
        "dataset": dataset,
        "src_tokenizer_path": str(src_tokenizer_path),
        "tgt_tokenizer_path": str(tgt_tokenizer_path),
        "max_sequence_length": src_tokenizer.max_length,
        "dataset_train_path": str(bundle.train_path),
        "dataset_valid_path": str(bundle.valid_path),
    }
    log.info("Hyperparameters initialised.", hyperparams=logging_hyperparameters)

    logger = TensorBoardLogger(LOGGING_DIR, name=f"{bundle.task_name}/{task.name}")
    logger.log_hyperparams(params=logging_hyperparameters)

    profiler = None
    if prod:
        # profiling will slow prod down.
        profiler = AdvancedProfiler(dirpath=logger.log_dir, filename=LOGGING_PERF_NAME)

    trainer = pl.Trainer(
        accelerator=accelerator,
        devices=devices,
        max_epochs=hyperparameters["n_epochs"],
        log_every_n_steps=log_every_n_steps,
        logger=logger,
        profiler=profiler,
        precision=precision,
        limit_train_batches=limit_train_batches,
        limit_val_batches=limit_val_batches,
        val_check_interval=val_check_interval,
        gradient_clip_val=0.5,
        gradient_clip_algorithm="norm",
        accumulate_grad_batches=task.kwargs["accumulate_grad_batches"],
        callbacks=[
            LearningRateMonitor(logging_interval="step"),
            ModelCheckpoint(save_top_k=3, monitor="val_loss"),
        ],
    )

    trainer.fit(
        task, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader
    )
    return trainer


def _int_or_float(value: str) -> Union[int, float]:
    """Keep the literal type: ``1`` is one batch, ``1.0`` is a full epoch.

    Lightning treats ints as batch counts and floats as epoch fractions, so
    parsing must not collapse ``1.0`` into ``1``.
    """
    value = value.strip()
    return int(value) if value.isdigit() else float(value)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train a machine-translation or publications model.",
    )
    parser.add_argument("--model", choices=[RNN, TRANSFORMER], default=TRANSFORMER)
    parser.add_argument("--dataset", choices=DATASETS, default="dummy")
    parser.add_argument(
        "--tokenizer",
        choices=TOKENIZER_OPTIONS,
        default="spm",
        help="bilingual tokenizer (ignored for --dataset dummy)",
    )
    parser.add_argument(
        "--prod",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="use the full-size model configuration (independent of the device)",
    )
    parser.add_argument("--legacy-transformer", action="store_true")
    parser.add_argument(
        "--back-translation",
        action="store_true",
        help="swap the source/target languages",
    )
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--max-seq-len", type=int, default=None)
    parser.add_argument("--max-epochs", type=int, default=None)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--precision", default="16-mixed")
    parser.add_argument(
        "--accelerator", choices=["auto", "cpu", "mps", "cuda"], default="auto"
    )
    parser.add_argument(
        "--devices",
        default="auto",
        help='number of devices or "auto"',
    )
    parser.add_argument(
        "--limit-train-batches",
        type=_int_or_float,
        default=1.0,
        help="batches (int, e.g. 1) or fraction of an epoch (float, e.g. 0.5)",
    )
    parser.add_argument(
        "--limit-val-batches",
        type=_int_or_float,
        default=1.0,
        help="batches (int) or fraction of the validation set (float)",
    )
    parser.add_argument(
        "--val-check-interval",
        type=_int_or_float,
        default=1.0,
        help="steps (int) or fraction of an epoch (float); default: once per epoch",
    )
    parser.add_argument("--log-every-n-steps", type=int, default=20)
    parser.add_argument("--n-workers", type=int, default=7)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--train-path", type=Path, default=None)
    parser.add_argument("--valid-path", type=Path, default=None)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> pl.Trainer:
    args = build_parser().parse_args(argv)
    devices: Union[str, int] = (
        int(args.devices) if args.devices.isdigit() else args.devices
    )
    return train(
        model=args.model,
        dataset=args.dataset,
        tokenizer=args.tokenizer,
        prod=args.prod,
        legacy_transformer=args.legacy_transformer,
        back_translation=args.back_translation,
        batch_size=args.batch_size,
        max_seq_len=args.max_seq_len,
        n_epochs=args.max_epochs,
        learning_rate=args.learning_rate,
        precision=args.precision,
        accelerator=args.accelerator,
        devices=devices,
        limit_train_batches=args.limit_train_batches,
        limit_val_batches=args.limit_val_batches,
        val_check_interval=args.val_check_interval,
        log_every_n_steps=args.log_every_n_steps,
        n_workers=args.n_workers,
        seed=args.seed,
        train_path=args.train_path,
        valid_path=args.valid_path,
    )


if __name__ == "__main__":
    main()
