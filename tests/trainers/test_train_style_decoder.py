"""Smoke test for the style-decoder training entry point.

This lives here rather than in ``tests/models/test_style_decoder.py`` because
that module builds its BERT and Marian fixtures at import time and would pull a
Hub download into what should be a fast CLI check.
"""

from pathlib import Path

import lightning.pytorch as pl
import pytest
import torch

from specialk.core.constants import PROJECT_DIR
from specialk.core.utils import check_torch_device
from specialk.models.classifier.models import CNNClassifier
from specialk.models.tokenizer import SentencePieceVocabulary
from specialk.models.transformer.torch.pytorch_transformer import (
    PyTorchTransformerModule,
)
from specialk.trainers import train_style_decoder

STYLE_FIXTURE = (
    Path(__file__).parents[1] / "test_files" / "datasets" / "en_fr_labels.parquet"
)
SEQUENCE_LENGTH = 32


def save_checkpoint(module, path: Path) -> None:
    """Write the keys Lightning reloads from; no trainer or hparams file."""
    torch.save(
        {
            "state_dict": module.state_dict(),
            "hyper_parameters": dict(module.hparams),
            # ``load_from_checkpoint`` reads this before anything else.
            "pytorch-lightning_version": pl.__version__,
            "epoch": 0,
            "global_step": 0,
        },
        path,
    )


def make_seq2seq(tokenizer: SentencePieceVocabulary) -> PyTorchTransformerModule:
    return PyTorchTransformerModule(
        name="transformer_tiny",
        vocabulary_size=tokenizer.vocab_size,
        sequence_length=SEQUENCE_LENGTH,
        tokenizer=tokenizer,
        decoder_tokenizer=tokenizer,
        decoder_vocabulary_size=tokenizer.vocab_size,
        dim_model=32,
        n_heads=2,
        dim_feedforward=64,
        num_encoder_layers=1,
        num_decoder_layers=1,
    )


def make_cnn_classifier(tokenizer: SentencePieceVocabulary) -> CNNClassifier:
    return CNNClassifier(
        name="tiny",
        vocabulary_size=tokenizer.vocab_size,
        sequence_length=SEQUENCE_LENGTH,
    )


@pytest.mark.lightweight
def test_cli_smoke(tmp_path, monkeypatch):
    tokenizer = SentencePieceVocabulary.from_file(
        PROJECT_DIR / "assets" / "tokenizer" / "sentencepiece" / "enfr_small.model",
        max_length=SEQUENCE_LENGTH,
    )
    mt_checkpoint = tmp_path / "mt.ckpt"
    save_checkpoint(make_seq2seq(tokenizer), mt_checkpoint)
    classifier_checkpoint = tmp_path / "cnn.ckpt"
    save_checkpoint(make_cnn_classifier(tokenizer), classifier_checkpoint)

    # Keep TensorBoard events out of the repo.
    monkeypatch.setattr(
        "specialk.core.experiment.LOGGING_DIR", tmp_path / "tb_logs"
    )
    checkpoint_dir = tmp_path / "ckpt"

    trainer = train_style_decoder.main(
        [
            "--dataset",
            "political",
            "--target-label",
            "1",
            "--mt-checkpoint",
            str(mt_checkpoint),
            "--classifier-checkpoint",
            str(classifier_checkpoint),
            "--classifier",
            "cnn",
            "--batch-size",
            "4",
            "--max-epochs",
            "1",
            "--limit-train-batches",
            "1",
            "--limit-val-batches",
            "1",
            # Use whatever this machine has; MPS on the development machine.
            "--accelerator",
            check_torch_device(),
            "--precision",
            "32-true",
            "--n-workers",
            "0",
            "--seed",
            "1337",
            "--checkpoint-dir",
            str(checkpoint_dir),
            "--path",
            str(STYLE_FIXTURE),
            "--source-column",
            "text_fr",
            "--logger",
            "tensorboard",
        ]
    )

    assert trainer.global_step == 1
    assert list(checkpoint_dir.glob("*.ckpt")), "the trainer wrote no checkpoint"

    eval_loss = trainer.callback_metrics.get("eval_loss")
    assert eval_loss is not None, sorted(trainer.callback_metrics)
    assert torch.isfinite(eval_loss)


@pytest.mark.heavyweight
def test_marian_target_vocabulary_uses_target_spm():
    text = ["The senator voted against the bill."]
    vocab = train_style_decoder.MarianTargetVocabulary(
        name="marian",
        pretrained_model_name_or_path="Helsinki-NLP/opus-mt-fr-en",
        max_length=16,
    )
    expected = vocab.tokenizer(text_target=text, max_length=16, padding="max_length")
    assert vocab.to_tensor(text)[0].tolist() == expected["input_ids"][0]
