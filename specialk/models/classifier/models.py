from __future__ import division

from typing import List, Optional, Tuple, Union

import lightning.pytorch as pl
import torch
import torch.nn as nn
from jaxtyping import Int
from torch import Tensor

from specialk.core.utils import batch_texts, check_torch_device, log, namespace_to_dict
from specialk.datasets.dataloaders import (
    init_classification_dataloaders as init_dataloaders,
)
from specialk.models.classifier.onmt.CNNModels import ConvNet
from specialk.models.tokenizer import (
    BPEVocabulary,
    SentencePieceVocabulary,
    Vocabulary,
    WordVocabulary,
)

DEVICE: str = check_torch_device()


class TextClassifier(pl.LightningModule):
    """Base class for Text Classification."""

    def __init__(
        self,
        name: str,
        vocabulary_size: int,
        sequence_length: int,
        tokenizer: Optional[Vocabulary] = None,
    ):
        super().__init__()
        self.name = name
        self.vocabulary_size = vocabulary_size
        self.sequence_length = sequence_length
        self.criterion = nn.BCELoss()
        self.model = None
        self.tokenizer = tokenizer

    def validation_step(self, batch: dict, batch_idx: int):
        loss, acc = self._shared_eval_step(batch, batch_idx)
        metrics = {"val_acc": acc, "val_loss": loss}
        self.log_dict(metrics, batch_size=batch["text"].size(0))
        return metrics

    def test_step(self, batch: dict, batch_idx: int):
        loss, acc = self._shared_eval_step(batch, batch_idx)
        metrics = {"test_acc": acc, "test_loss": loss}
        self.log_dict(metrics)
        return metrics

    def predict_step(
        self, batch: dict, batch_idx: int, dataloader_idx=0
    ) -> torch.Tensor:
        raise NotImplementedError

    def _shared_eval_step(
        self, batch: dict, batch_idx: int
    ) -> Tuple[torch.Tensor, float]:
        """Run shared eval step.

        Args:
            batch (dict): individual batch generated from the DataLoader.
            batch_idx (int): ID corresponding to the batch.

        Returns:
            Tuple[torch.Tensor, float]: Returns loss, accuracy.
        """
        raise NotImplementedError

    @staticmethod
    def calculate_classification_metrics(
        output: torch.Tensor,
        target: torch.Tensor,
    ) -> float:
        """Calculate downstream metrics.

        Args:
            output (torch.Tensor): Predicted values generated from the model.
            target (torch.Tensor): Values we want to predict.

        Returns:
            int: Accuracy.
        """

        outputs = (output > 0.5).long()
        n_correct: float = outputs.eq(target).sum().item() / outputs.size(0)
        return n_correct


class CNNClassifier(TextClassifier):
    """CNN Text Classifier. Operates on one-hot."""

    def __init__(
        self,
        name: str,
        vocabulary_size: int,
        sequence_length: int,
        tokenizer: Optional[Vocabulary] = None,
    ):
        super().__init__(
            name=name,
            vocabulary_size=vocabulary_size,
            sequence_length=sequence_length,
            tokenizer=tokenizer,
        )
        # min_batch_size is used because the convolution
        # operations depend on a minimum size.
        self.min_batch_size = 50
        self.model = ConvNet(
            vocab_size=self.vocabulary_size, sequence_length=self.sequence_length
        )

    def training_step(self, batch: dict, batch_idx: int) -> torch.Tensor:
        """Run Training step.

        Args:
            batch (dict): individual batch generated from the DataLoader.
            batch_idx (int): ID corresponding to the batch.

        Returns:
            torch.Tensor: Returns loss.
        """
        x, y = batch["text"], batch["label"]
        batch_size, seq_len = x.size()

        if batch_size < self.min_batch_size:
            log.error(
                f"batch size (currently set to {batch_size}) "
                f"should be at least {self.min_batch_size}."
            )

        one_hot = self.one_hot(x)

        y_hat = self.model(one_hot).squeeze(-1)

        loss: torch.Tensor = self.criterion(y_hat, y.float())
        accuracy = self.calculate_classification_metrics(y_hat, y)

        self.log_dict(
            {"train_acc": accuracy, "batch_id": batch_idx, "train_loss": loss},
            batch_size=batch_size,
        )
        return loss

    def one_hot(self, text: Int[torch.Tensor, "batch seq_len"]) -> torch.Tensor:
        """wrap into one-hot encoding of tokens for activation."""
        batch_size, seq_length = text.shape
        return torch.zeros(
            seq_length, batch_size, self.vocabulary_size, device=self.device
        ).scatter_(2, torch.unsqueeze(text.T, 2), 1)

    def _shared_eval_step(
        self, batch: dict, batch_idx: int
    ) -> Tuple[torch.Tensor, float]:
        """Run shared eval step.

        Args:
            batch (dict): individual batch generated from the DataLoader.
            batch_idx (int): ID corresponding to the batch.

        Returns:
            Tuple[torch.Tensor, float]: Returns loss, accuracy.
        """
        x, y = batch["text"], batch["label"]

        one_hot = self.one_hot(x)

        y_hat = self.model(one_hot).squeeze(-1)
        loss: torch.Tensor = self.criterion(y_hat, y.float())

        accuracy = self.calculate_classification_metrics(y_hat, y)
        return loss, accuracy

    def validation_step(self, batch: dict, batch_idx: int):
        loss, acc = self._shared_eval_step(batch, batch_idx)
        metrics = {"val_acc": acc, "val_loss": loss}
        self.log_dict(metrics, batch_size=batch["text"].size(0))
        return metrics

    def test_step(self, batch: dict, batch_idx: int):
        loss, acc = self._shared_eval_step(batch, batch_idx)
        metrics = {"test_acc": acc, "test_loss": loss}
        self.log_dict(metrics)
        return metrics

    @torch.inference_mode()
    def generate(self, texts: list[str], batch_size: Optional[int] = 64) -> Tensor:
        batch: List[str]
        results = []
        for batch in batch_texts(texts, batch_size):
            tokens = self.tokenizer.to_tensor(batch)
            one_hot = self.one_hot(tokens)
            result = self.model(one_hot)
            results.append(result)
        return torch.cat(results).cpu()

    def configure_optimizers(self) -> torch.optim.Optimizer:
        return torch.optim.Adam(self.model.parameters(), lr=0.02)
