"""
Style Decoder Model.
"""

from typing import Dict

import lightning.pytorch as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from jaxtyping import Float, Int
from torch import Tensor
from torch.utils.data import DataLoader

from specialk.core.constants import SOURCE, TARGET
from specialk.models.classifier.trainer import TextClassifier
from specialk.models.mt_model import NMTModule
from specialk.models.ops.ops import token_accuracy


class StyleBackTranslationModel(pl.LightningModule):
    def __init__(
        self,
        mt_model: NMTModule,
        classifier: TextClassifier,
        smoothing: bool = True,
        use_mapping=True,
    ):
        """
        Args:
            mt_model (NMTModel):
                Machine Translation model (with target language to english).
            classifier (CNNModels):
                Style classifier model.
            smoothing (bool, optional):
                If set, adds smothing to reconstruction loss function.
                Defaults to True.
        """
        super().__init__()
        self.nmt_model: NMTModule = mt_model
        self.classifier: TextClassifier = classifier
        # the mapping matrix is learnt as we fine-tune the decoder model.
        self.mapping = (
            nn.Parameter(
                torch.zeros(
                    (self.nmt_model.vocabulary_size, self.classifier.vocabulary_size)
                )
            )
            if use_mapping
            else None
        )
        self.target_label: int = self.classifier.refs.tgt_label

        # encoder will always be in eval mode.
        # We're only updating the decoder weights.
        self.classifier.eval()

        # loss functions.
        self.criterion_class = nn.BCELoss()
        self.criterion_recon = torch.nn.CrossEntropyLoss(
            ignore_index=self.nmt_model.PAD
        )
        self.smoothing = smoothing

    def training_step(self, batch: dict, batch_idx: int) -> torch.Tensor:
        """
        Forward pass of the transformer.
        The output from the model is then sent to the classifier.

        Returns:
            Loss value to run gradients against.
        """

        x: Int[Tensor, "batch seq_len"] = batch[SOURCE]
        y: Int[Tensor, "batch seq_len"] = batch[TARGET]
        label: Int[Tensor, "batch"] = batch["class"]

        y_hat_text: Float[Tensor, "batch length vocab"]
        y_hat_text = self.nmt_model.model(x, y)

        if self.mapping is not None:
            # tokenizers for the models are different so
            # we need to project that over.
            y_hat_text: Float[Tensor, "batch length vocab_classifier"]
            y_hat_text = y_hat_text @ self.mapping

        y_hat_label: Float[Tensor, "batch"]
        y_hat_label = self.classifier(y_hat_text)

        # Calculate losses.
        loss_class = self.classifier_loss(y_hat_label, label)
        loss_reconstruction = self.reconstruction_loss(y_hat_text, y)
        joint_loss = loss_class + loss_reconstruction

        # report metrics.
        self.log_dict(
            {
                "train_acc": token_accuracy(y_hat_text, y, self.constants.PAD),
                "batch_id": batch_idx,
                "train_joint_loss": joint_loss,
                "train_recon_loss": loss_reconstruction,
                "train_class_loss": loss_class,
            }
        )
        return joint_loss

    def _shared_eval_step(self, batch: DataLoader, batch_idx: int):
        """Evaluation step used for both eval/test runs on a given dataset.

        Args:
            batch (DataLoader): _description_
            batch_idx (int): _description_

        Returns:
            _type_: _description_
        """
        x: Int[Tensor, "batch seq_len"] = batch[SOURCE]
        y: Int[Tensor, "batch seq_len"] = batch[TARGET]
        label: Int[Tensor, "batch"] = batch["class"]

        y_hat_text: Float[Tensor, "batch length vocab"]
        y_hat_text = self.mt_forward(x, y)

        y_hat_label: Float[Tensor, "batch"]
        y_hat_label = self.classifier(y_hat_text)

        # Calculate losses.
        loss_class = self.classifier_loss(y_hat_label, label)
        loss_reconstruction = self.reconstruction_loss(y_hat_text, y)
        joint_loss = loss_class + loss_reconstruction
        accuracy = token_accuracy(y_hat_text, y, self.constants.PAD)

        return loss_reconstruction, loss_class, joint_loss, accuracy

    def validation_step(self, batch: dict, batch_idx: int) -> Dict[str, torch.Tensor]:
        """
        Validation run.
        """
        l_recon, l_class, l_joint, accuracy = self._shared_eval_step(batch, batch_idx)
        metrics: Dict[str, torch.Tensor] = {
            "valid_accuracy": accuracy,
            "valid_joint_loss": l_joint,
            "valid_class_loss": l_class,
            "valid_recon_loss": l_recon,
        }
        self.log_dict(metrics)
        return metrics

    def test_step(self, batch: dict, batch_idx: int) -> Dict[str, torch.Tensor]:
        """Test run."""
        l_recon, l_class, l_joint, accuracy = self._shared_eval_step(batch, batch_idx)
        metrics: Dict[str, torch.Tensor] = {
            "test_accuracy": accuracy,
            "test_joint_loss": l_joint,
            "test_class_loss": l_class,
            "test_recon_loss": l_recon,
        }
        self.log_dict(metrics)
        return metrics

    def classifier_loss(
        self, pred: Float[Tensor, "batch 1"], ref: Int[Tensor, "batch"]
    ) -> Tensor:
        """
        Computes classifier loss.
        """
        return self.classifier.criterion(pred, ref)

    def reconstruction_loss(
        self, pred: Float[Tensor, "batch seq vocab"], ref: Int[Tensor, "batch seq"]
    ) -> Tensor:
        """
        Computes reconstruction loss.
        """
        return self.nmt_model.criterion(pred.view(-1, pred.size(-1)), ref.view(-1))

    def configure_optimizers(self):
        # return torch.optim.Adam(self.model.parameters(), lr=0.02)
        return None
