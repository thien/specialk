"""
Style Decoder Model.
"""

from typing import Dict, Optional, Tuple

import lightning.pytorch as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from jaxtyping import Float, Int
from torch import Tensor
from torch.utils.data import DataLoader

from specialk.core import log
from specialk.core.constants import PAD, SOURCE, TARGET
from specialk.models.classifier.trainer import (
    BERTClassifier,
    CNNClassifier,
    TextClassifier,
)
from specialk.models.mt_model import NMTModule
from specialk.models.ops.ops import token_accuracy


class StyleBackTranslationModel(pl.LightningModule):
    def __init__(
        self,
        mt_model: NMTModule,
        classifier: TextClassifier,
        target_label: int = 1,
        use_mapping=True,
        mapping_dim: int = 256,
    ):
        """
        Args:
            mt_model (NMTModel):
                Machine Translation model (with target language to english).
            classifier (CNNModels):
                Style classifier model.
            mapping_dim (int):
                Dimension to learn the mapping between vocabularies
                from the sequence model to the classifier model.
        """
        super().__init__()
        self.sequence_model: NMTModule = mt_model
        self.classifier: TextClassifier = classifier
        # the mapping matrix is learnt as we fine-tune the decoder model.
        self.use_mapping = use_mapping

        # having one tensor is an unreaonable size; (marianMT is 50k, BERT 30k)...
        self.seq_projection = (
            nn.Linear(self.sequence_model.vocabulary_size, mapping_dim)
            if self.use_mapping
            else None
        )
        self.clf_projection = (
            nn.Linear(mapping_dim, self.classifier.vocabulary_size)
            if self.use_mapping
            else None
        )

        self.target_label: int = target_label

        # encoder will always be in eval mode.
        self.classifier.eval()

    def training_step(self, batch: dict, batch_idx: int) -> torch.Tensor:
        """
        Forward pass of the transformer.
        The output from the model is then sent to the classifier.

        Returns:
            Loss value to run gradients against.
        """

        x: Int[Tensor, "batch seq_len"] = batch[SOURCE].squeeze(1)
        y: Int[Tensor, "batch seq_len"] = batch[TARGET].squeeze(1)
        y_label: Int[Tensor, "batch"] = batch["label"]

        log.info("x.shape", xhape=x.shape)

        y_hat_text, y_hat_label = self._forward(x, y, y_label)

        # Calculate losses.
        y_hat_text = y_hat_text.view(-1, y_hat_text.size(-1))
        y = y.view(-1)
        loss_reconstruction = self.reconstruction_loss(y_hat_text, y)
        loss_class = y_hat_label.loss
        joint_loss = loss_class + loss_reconstruction

        # report metrics.
        self.log_dict(
            {
                "train_acc": token_accuracy(y_hat_text, y, PAD),
                "batch_id": batch_idx,
                "train_joint_loss": joint_loss,
                "train_recon_loss": loss_reconstruction,
                "train_class_loss": loss_class,
            }
        )
        return joint_loss

    def _forward(
        self,
        x: Int[Tensor, "batch seq_len"],
        y: Int[Tensor, "batch seq_len"],
        y_label: Optional[Int[Tensor, "batch"]] = None,
    ) -> Tuple[Tensor, Optional[Tensor]]:
        """Performs forward pass on the mt model and the discriminator model."""
        y_hat_text: Float[Tensor, "batch length vocab"]
        y_hat_text = self.sequence_model(x, y)

        if self.use_mapping:
            # tokenizers for the models are different so
            # we need to project that over.
            y_hat_text: Float[Tensor, "batch length vocab_classifier"]
            y_hat_text = self.seq_projection(y_hat_text)
            y_hat_text = self.clf_projection(y_hat_text)

        y_hat_label: Float[Tensor, "batch"]
        y_hat_label = self.classifier(y_hat_text)
        return y_hat_text, y_hat_label

    def _shared_eval_step(self, batch: dict, batch_idx: int):
        """Evaluation step used for both eval/test runs on a given dataset.

        Args:
            batch (DataLoader): _description_
            batch_idx (int): _description_

        Returns:
            _type_: _description_
        """
        x: Int[Tensor, "batch seq_len"] = batch[SOURCE].squeeze(1)
        y: Int[Tensor, "batch seq_len"] = batch[TARGET].squeeze(1)
        y_label: Int[Tensor, "batch"] = batch["class"]

        y_hat_text, y_hat_label = self._forward(x, y, y_label)

        # Calculate losses.
        y_hat_text = y_hat_text.view(-1, y_hat_text.size(-1))
        y = y.view(-1)
        loss_reconstruction = self.reconstruction_loss(y_hat_text, y)
        loss_class = y_hat_label.loss
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

    def reconstruction_loss(self, y_hat, y) -> Tensor:
        return self.sequence_model.criterion(y_hat, y)

    def classifier_loss(self, y_hat, y) -> Tensor:
        return self.classifier.criterion(y_hat, y)

    def _get_lora_params(self):
        """Filter and return only the LoRA parameters."""
        lora_params = []
        for name, param in self.sequence_model.model.named_parameters():
            # Assuming LoRA parameters have 'lora_' in their name
            if "lora_" in name:
                lora_params.append(param)
        return lora_params

    def configure_optimizers(self):
        params = self._get_lora_params()
        if len(params) == 0:
            params = list(self.sequence_model.model.parameters())

        if self.use_mapping:
            params += list(self.clf_projection.parameters())
            params += list(self.seq_projection.parameters())
        return torch.optim.AdamW(params, lr=0.002)


class StyleBackTranslationModelWithCNN(StyleBackTranslationModel):
    def __init__(
        self,
        mt_model: NMTModule,
        classifier: CNNClassifier,
        **kwargs,
    ):
        assert isinstance(
            classifier, CNNClassifier
        ), f"Classifier is not a CNNClassifier (found {type(classifier).__name__})"
        super().__init__(mt_model=mt_model, classifier=classifier, **kwargs)

    def _forward(
        self,
        x: Int[Tensor, "batch seq_len"],
        y: Int[Tensor, "batch seq_len"],
        y_label: Optional[Int[Tensor, "batch"]] = None,
    ) -> Tuple[Tensor, Optional[Tensor]]:
        y_hat_text: Float[Tensor, "batch length vocab_seq2seq"]
        y_hat_text = self.sequence_model(x, y)

        classifier_text: Float[Tensor, "seq_len batch d_vocab_classifier"]
        classifier_text = y_hat_text.transpose(0, 1)
        if self.use_mapping:
            classifier_text = self.seq_projection(classifier_text)
            classifier_text = self.clf_projection(classifier_text)

        # classifer pass
        y_hat_label: Float[Tensor, "batch length vocab"]
        y_hat_label = self.classifier.model(classifier_text).squeeze(-1)
        return y_hat_text, y_hat_label


class StyleBackTranslationModelWithBERT(StyleBackTranslationModel):
    def __init__(self, mt_model: NMTModule, classifier: BERTClassifier, **kwargs):
        assert isinstance(
            classifier, BERTClassifier
        ), f"Classifier is not a BERTClassifier (found {type(classifier).__name__})"
        super().__init__(
            mt_model=mt_model, classifier=classifier, use_mapping=True, **kwargs
        )

    def _forward(
        self,
        x: Int[Tensor, "batch seq_len"],
        y: Int[Tensor, "batch seq_len"],
        y_label: Optional[Int[Tensor, "batch"]] = None,
    ) -> Tuple[Tensor, Optional[Tensor]]:
        y_hat_text: Float[Tensor, "batch length vocab"]
        y_hat_text = self.sequence_model(x, y)

        # tokenizers for the models are going to be different. We're going to
        # learn the mapping as we go.

        # tokenizers for the models are different so
        # we need to project that over.
        y_hat_text: Float[Tensor, "batch length vocab_classifier"]
        y_hat_text = self.seq_projection(y_hat_text)
        y_hat_text = self.clf_projection(y_hat_text)
        y_hat_embs = (
            y_hat_text @ self.classifier.model.bert.embeddings.word_embeddings.weight
        )

        y_hat_mask = torch.argmax(y_hat_text, dim=-1) == PAD

        y_hat_label: Float[Tensor, "batch"]
        y_hat_label = self.classifier.model(
            inputs_embeds=y_hat_embs, attention_mask=y_hat_mask, labels=y_label
        )
        return y_hat_text, y_hat_label
