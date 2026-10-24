"""Learning check for the style decoder.

Overfit one fixed batch and watch the numbers move: the classification loss
falls, the validation style accuracy on that batch goes up, and the
reconstruction loss stays bounded. It runs in seconds on MPS.

    .venv/bin/python -m pytest tests/trainers/test_style_decoder_learns.py \\
        -m benchmark -s
"""

import math
from pathlib import Path

import lightning.pytorch as pl
import pytest
from lightning.pytorch.callbacks import Callback
from torch.utils.data import DataLoader

from datasets import Dataset
from specialk.core.constants import PROJECT_DIR
from specialk.core.utils import check_torch_device
from specialk.models.classifier.models import CNNClassifier
from specialk.models.style_decoder import StyleBackTranslationModelWithCNN
from specialk.models.tokenizer import SentencePieceVocabulary
from specialk.models.transformer.torch.pytorch_transformer import (
    PyTorchTransformerModule,
)

FIXTURE = Path(__file__).parents[1] / "test_files" / "datasets" / "en_fr_labels.parquet"
TOKENIZER_PATH = (
    PROJECT_DIR / "assets" / "tokenizer" / "sentencepiece" / "enfr_small.model"
)

BATCH_SIZE = 50
SEQUENCE_LENGTH = 40
STEPS = 200
SEED = 1337


class LossRecorder(Callback):
    """Collect the class and reconstruction losses, one value per step."""

    def __init__(self):
        super().__init__()
        self.class_losses: list[float] = []
        self.recon_losses: list[float] = []

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        metrics = trainer.callback_metrics
        self.class_losses.append(float(metrics["train_class_loss"]))
        self.recon_losses.append(float(metrics["train_recon_loss"]))


@pytest.fixture(scope="module")
def tokenizer() -> SentencePieceVocabulary:
    return SentencePieceVocabulary.from_file(TOKENIZER_PATH, max_length=SEQUENCE_LENGTH)


@pytest.fixture(scope="module")
def fixed_batch(tokenizer: SentencePieceVocabulary) -> DataLoader:
    """A fixed batch of fifty rows.

    The fixture has ten labelled rows, so I repeat them five times. Overfitting
    a single batch is the point; more rows would only make the run longer.
    """
    dataset = Dataset.from_parquet(str(FIXTURE)).class_encode_column("label")
    dataset = dataset.select([i % len(dataset) for i in range(BATCH_SIZE)])

    def tokenize(example):
        example["source"] = tokenizer.to_tensor(example["text_fr"])
        example["target"] = tokenizer.to_tensor(example["text"])
        return example

    dataset = dataset.map(tokenize).select_columns(["source", "target", "label"])
    return DataLoader(dataset.with_format("torch"), batch_size=BATCH_SIZE)


def build_module(
    tokenizer: SentencePieceVocabulary,
) -> StyleBackTranslationModelWithCNN:
    seq2seq = PyTorchTransformerModule(
        name="tiny",
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
    classifier = CNNClassifier(
        name="tiny",
        vocabulary_size=tokenizer.vocab_size,
        sequence_length=SEQUENCE_LENGTH,
    )
    return StyleBackTranslationModelWithCNN(
        seq2seq,
        classifier,
        target_label=1,
        # No vocabulary mapping here, so the decoder alone has to beat the recon loss.
        lambda_class=3.0,
        tau_start=1.0,
        tau_end=0.1,
        learning_rate=1e-3,
    )


def evaluate(trainer: pl.Trainer, module, loader: DataLoader) -> float:
    """Run validation on the fixed batch and return the style accuracy."""
    trainer.validate(module, dataloaders=loader, verbose=False)
    return float(trainer.callback_metrics["eval_style_acc"])


@pytest.mark.benchmark
def test_overfits_one_batch(fixed_batch, tokenizer):
    pl.seed_everything(SEED, workers=True)
    module = build_module(tokenizer)

    recorder = LossRecorder()
    trainer = pl.Trainer(
        max_epochs=STEPS,
        accelerator=check_torch_device(),
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=False,
        enable_model_summary=False,
        limit_train_batches=1,
        num_sanity_val_steps=0,
        gradient_clip_val=0.5,
        gradient_clip_algorithm="norm",
        callbacks=[recorder],
    )

    initial_acc = evaluate(trainer, module, fixed_batch)

    # The untrained head gives every row the same label, and which side of 0.5
    # it lands on is luck. Train toward the other label, so accuracy starts at
    # zero and has somewhere to go.
    if initial_acc == 1.0:
        module.target_label = 1 - module.target_label
        initial_acc = evaluate(trainer, module, fixed_batch)

    trainer.fit(module, train_dataloaders=fixed_batch)
    final_acc = evaluate(trainer, module, fixed_batch)

    first_class_loss = recorder.class_losses[0]
    last_class_loss = recorder.class_losses[-1]
    first_recon_loss = recorder.recon_losses[0]
    last_recon_loss = recorder.recon_losses[-1]

    print(
        f"\nclass loss {first_class_loss:.4f} -> {last_class_loss:.4f}, "
        f"style acc {initial_acc:.3f} -> {final_acc:.3f}, "
        f"recon loss {first_recon_loss:.4f} -> {last_recon_loss:.4f}"
    )

    assert len(recorder.class_losses) == STEPS
    assert last_class_loss < 0.8 * first_class_loss
    assert final_acc > initial_acc
    assert final_acc == 1.0
    assert all(math.isfinite(x) for x in recorder.recon_losses)
    assert max(recorder.recon_losses) < 2 * first_recon_loss

    # The classifier is frozen and in eval mode throughout.
    assert not module.classifier.training
    assert all(not p.requires_grad for p in module.classifier.parameters())
    assert all(p.grad is None for p in module.classifier.parameters())
