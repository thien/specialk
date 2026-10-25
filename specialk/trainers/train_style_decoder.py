"""Training entry point for the style decoder.

Fine-tunes a saved seq2seq checkpoint so that its translations are classified
as the target style. Training updates the decoder and the vocabulary mapping.
The encoder and classifier stay frozen.

Pass ``--seq2seq marian`` for a HuggingFace Marian checkpoint. ``--lora-r``
then trains a LoRA adapter on the decoder's cross-attention modules.

Example:
    python -m specialk.trainers.train_style_decoder --dataset political \\
        --target-label 1 --mt-checkpoint checkpoints/mt.ckpt \\
        --classifier-checkpoint checkpoints/cnn.ckpt --classifier cnn \\
        --limit-train-batches 2 --limit-val-batches 1
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

import lightning.pytorch as pl
from lightning.pytorch.callbacks import ModelCheckpoint
from peft import LoraConfig, get_peft_model
from peft.utils.peft_types import TaskType
from torch import Tensor

from specialk.core.constants import SEED, VAL_LOSS
from specialk.core.experiment import add_logger_args, init_logger
from specialk.core.utils import log
from specialk.models.classifier.models import BERTClassifier, CNNClassifier
from specialk.models.style_decoder import (
    StyleBackTranslationModel,
    StyleBackTranslationModelWithBERT,
    StyleBackTranslationModelWithCNN,
)
from specialk.models.tokenizer import HuggingFaceVocabulary
from specialk.models.transformer.hf_transformer import MarianMTModule
from specialk.models.transformer.torch.pytorch_transformer import (
    PyTorchTransformerModule,
)
from specialk.trainers.common import (
    STYLE_DATASETS,
    init_dataloader,
    load_style_datasets,
)
from specialk.trainers.train import _int_or_float

DATASETS = tuple(sorted(STYLE_DATASETS))
CLASSIFIERS = ("cnn", "bert")
SEQ2SEQ = ("transformer", "marian")


def load_seq2seq(checkpoint_path: str | Path) -> PyTorchTransformerModule:
    """Load the seq2seq checkpoint whose tokenizer tokenizes the data.

    The checkpoint holds a ``Vocabulary`` object, which ``torch.load`` refuses
    by default, so load it with ``weights_only=False``. It is a local file the
    caller pointed us at.
    """
    return PyTorchTransformerModule.load_from_checkpoint(
        checkpoint_path, map_location="cpu", weights_only=False
    )


def load_marian(
    checkpoint_path: str | Path, lora_r: int | None = None
) -> MarianMTModule:
    """Load a Marian checkpoint and attach LoRA when ``lora_r`` is set.

    A checkpoint that already carries an adapter keeps it; ``--lora-r`` then
    has to match the adapter's rank. A checkpoint without adapters gets a new
    one on the decoder's cross-attention modules.
    """
    model = MarianMTModule.load_from_checkpoint(
        checkpoint_path, map_location="cpu", weights_only=False
    )
    if lora_r is None:
        return model

    if model.peft_config is not None:
        if model.peft_config.r != lora_r:
            raise ValueError(
                f"The checkpoint has a LoRA adapter of rank "
                f"{model.peft_config.r}, but --lora-r is {lora_r}"
            )
        return model

    peft_config = LoraConfig(
        task_type=TaskType.SEQ_2_SEQ_LM,
        r=lora_r,
        lora_alpha=4 * lora_r,
        lora_dropout=0.1,
        bias="none",
        target_modules=model.get_lora_config_modules(),
    )
    model.model = get_peft_model(model.model, peft_config, adapter_name=model.name)
    model.peft_config = peft_config
    model._freeze_non_lora_weights()
    log.info("Attached a LoRA adapter to the Marian decoder", rank=lora_r)
    return model


class MarianTargetVocabulary(HuggingFaceVocabulary):
    """Encodes English targets with Marian's target sentencepiece model."""

    def to_tensor(self, text: str | list[str]) -> Tensor:
        return self.tokenizer(
            text_target=text,
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )["input_ids"]


def marian_tokenizers(
    model: MarianMTModule,
) -> tuple[HuggingFaceVocabulary, MarianTargetVocabulary]:
    """Source and target tokenizers that ``init_dataloader`` can call ``to_tensor`` on."""
    options = {
        "name": model.name,
        "pretrained_model_name_or_path": model.model_base_name,
        "max_length": model.sequence_length,
    }
    return HuggingFaceVocabulary(**options), MarianTargetVocabulary(**options)


def load_classifier(
    kind: str, checkpoint_path: str | Path
) -> CNNClassifier | BERTClassifier:
    """Load a trained style classifier of the requested kind."""
    if kind == "cnn":
        return CNNClassifier.load_from_checkpoint(checkpoint_path, map_location="cpu")
    if kind == "bert":
        return BERTClassifier.load_from_checkpoint(checkpoint_path, map_location="cpu")
    raise ValueError(f"Unknown classifier {kind!r}; expected one of {CLASSIFIERS}")


def build_style_model(
    mt_model: PyTorchTransformerModule | MarianMTModule,
    classifier: CNNClassifier | BERTClassifier,
    target_label: int,
    lambda_class: float,
    tau_start: float,
    tau_end: float,
    learning_rate: float,
) -> StyleBackTranslationModel:
    """Wrap the two frozen models in the matching style decoder."""
    options = {
        "target_label": target_label,
        "lambda_class": lambda_class,
        "tau_start": tau_start,
        "tau_end": tau_end,
        "learning_rate": learning_rate,
    }
    if isinstance(classifier, CNNClassifier):
        return StyleBackTranslationModelWithCNN(mt_model, classifier, **options)
    return StyleBackTranslationModelWithBERT(mt_model, classifier, **options)


def train(args: argparse.Namespace) -> pl.Trainer:
    """Train the style decoder described by the parsed CLI arguments."""
    if args.lora_r is not None and args.seq2seq != "marian":
        raise ValueError("--lora-r only applies to --seq2seq marian")

    pl.seed_everything(args.seed, workers=True)

    if args.seq2seq == "marian":
        mt_model = load_marian(args.mt_checkpoint, args.lora_r)
        # Marian's checkpoint records a vocabulary size of 1, so read the real
        # one from its tokenizer before the style decoder builds its projections.
        mt_model.vocabulary_size = mt_model.tokenizer.vocab_size
        tokenizer, decoder_tokenizer = marian_tokenizers(mt_model)
        sequence_length = mt_model.sequence_length
    else:
        mt_model = load_seq2seq(args.mt_checkpoint)
        tokenizer, decoder_tokenizer = mt_model.tokenizer, mt_model.decoder_tokenizer
        if tokenizer is None:
            raise ValueError(
                f"The seq2seq checkpoint {args.mt_checkpoint!r} holds no tokenizer; "
                "pass one saved by specialk.trainers.train"
            )
        # ``to_tensor`` pads to the tokenizer's own max length, so the dataloaders
        # need no separate sequence length; report it in the run log.
        sequence_length = (
            getattr(tokenizer, "max_length", None) or mt_model.sequence_length
        )
    classifier = load_classifier(args.classifier, args.classifier_checkpoint)

    train_dataset, val_dataset = load_style_datasets(
        args.dataset,
        target_label=args.target_label,
        num_proc=max(args.n_workers, 1),
        path=args.path,
        source_column=args.source_column,
    )
    if len(train_dataset) == 0:
        raise ValueError(
            f"No {args.dataset!r} rows have label {args.target_label}; "
            "check --target-label and --source-column"
        )

    train_dataloader, _ = init_dataloader(
        train_dataset,
        tokenizer,
        args.batch_size,
        shuffle=True,
        decoder_tokenizer=decoder_tokenizer,
        n_workers=args.n_workers,
    )
    val_dataloader, _ = init_dataloader(
        val_dataset,
        tokenizer,
        args.batch_size,
        shuffle=False,
        decoder_tokenizer=decoder_tokenizer,
        n_workers=args.n_workers,
    )

    task = build_style_model(
        mt_model,
        classifier,
        target_label=args.target_label,
        lambda_class=args.lambda_class,
        tau_start=args.tau_start,
        tau_end=args.tau_end,
        learning_rate=args.learning_rate,
    )

    log.info(
        "Loaded the style-decoder setup",
        dataset=args.dataset,
        target_label=args.target_label,
        seq2seq=args.seq2seq,
        lora_r=args.lora_r,
        mt_checkpoint=str(args.mt_checkpoint),
        classifier_checkpoint=str(args.classifier_checkpoint),
        classifier=args.classifier,
        train_rows=len(train_dataset),
        val_rows=len(val_dataset),
        tokenizer=tokenizer.__class__.__name__,
        sequence_length=sequence_length,
        vocabulary_size=mt_model.vocabulary_size,
        use_mapping=task.use_mapping,
        mapping_coverage=getattr(task, "mapping_coverage", None),
    )

    params = {
        "batch_size": args.batch_size,
        "dataset": args.dataset,
        "target_label": args.target_label,
        "seq2seq": args.seq2seq,
        "lora_r": args.lora_r,
        "mt_checkpoint": str(args.mt_checkpoint),
        "classifier_checkpoint": str(args.classifier_checkpoint),
        "classifier": args.classifier,
        "lambda_class": args.lambda_class,
        "tau_start": args.tau_start,
        "tau_end": args.tau_end,
        "learning_rate": args.learning_rate,
        "max_epochs": args.max_epochs,
        "use_mapping": task.use_mapping,
        "mapping_coverage": getattr(task, "mapping_coverage", None),
    }
    logger = init_logger(
        args, f"style_decoder/{args.dataset}_{args.target_label}", params
    )

    checkpoint_dir = (
        Path(args.checkpoint_dir)
        if args.checkpoint_dir
        else Path.cwd() / "checkpoints"
    )

    trainer = pl.Trainer(
        accelerator=args.accelerator,
        max_epochs=args.max_epochs,
        logger=logger,
        precision=args.precision,
        limit_train_batches=args.limit_train_batches,
        limit_val_batches=args.limit_val_batches,
        gradient_clip_val=0.5,
        gradient_clip_algorithm="norm",
        callbacks=[
            # Every epoch is kept because the thesis found three before
            # overfitting.
            ModelCheckpoint(
                dirpath=checkpoint_dir, monitor=VAL_LOSS, save_top_k=-1
            ),
        ],
    )

    trainer.fit(
        task, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader
    )
    return trainer


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Train the style decoder on top of a frozen seq2seq checkpoint. "
            "Use --seq2seq marian for a HuggingFace Marian checkpoint, and "
            "--lora-r to train a LoRA adapter on its decoder."
        ),
    )
    parser.add_argument("--dataset", choices=DATASETS, default="political")
    parser.add_argument(
        "--target-label",
        type=int,
        default=1,
        help="label the translations should be classified as",
    )
    parser.add_argument(
        "--mt-checkpoint",
        type=Path,
        required=True,
        help="seq2seq checkpoint whose tokenizer and decoder are used",
    )
    parser.add_argument(
        "--seq2seq",
        choices=SEQ2SEQ,
        default="transformer",
        help="kind of seq2seq checkpoint --mt-checkpoint points at",
    )
    parser.add_argument(
        "--lora-r",
        type=int,
        default=None,
        help=(
            "LoRA rank for the Marian decoder; requires --seq2seq marian and "
            "must match a rank the checkpoint already carries"
        ),
    )
    parser.add_argument(
        "--classifier-checkpoint",
        type=Path,
        required=True,
        help="trained style classifier checkpoint",
    )
    parser.add_argument("--classifier", choices=CLASSIFIERS, default="cnn")
    parser.add_argument("--lambda-class", type=float, default=1.0)
    parser.add_argument("--tau-start", type=float, default=1.0)
    parser.add_argument("--tau-end", type=float, default=0.1)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-epochs", type=int, default=3)
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
        "--accelerator", choices=["auto", "cpu", "mps", "cuda"], default="auto"
    )
    parser.add_argument(
        "--precision",
        default="32-true",
        help="Lightning precision; MPS runs in full precision",
    )
    parser.add_argument("--n-workers", type=int, default=7)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        default=None,
        help="where to write checkpoints (default: ./checkpoints)",
    )
    parser.add_argument(
        "--path",
        type=Path,
        default=None,
        help="load a local parquet instead of the Hub dataset",
    )
    parser.add_argument(
        "--source-column",
        default=None,
        help=(
            "name of the French column; political defaults to text_fr, "
            "publications to fr_text"
        ),
    )
    add_logger_args(parser, "style-decoder")
    return parser


def main(argv: Sequence[str] | None = None) -> pl.Trainer:
    args = build_parser().parse_args(argv)
    return train(args)


if __name__ == "__main__":
    main()
