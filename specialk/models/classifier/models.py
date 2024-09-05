from __future__ import division

import tempfile
import warnings
from pathlib import Path
from typing import List, Optional, Tuple

import lightning.pytorch as pl
import safetensors
import torch
import torch.nn as nn
from jaxtyping import Float, Int
from peft import (
    LoraConfig,
    PeftModel,
    TaskType,
    get_peft_model,
    get_peft_model_state_dict,
)
from torch import Tensor
from transformers import AutoConfig, AutoModelForSequenceClassification, AutoTokenizer

from specialk.core.constants import (
    TEST_ACC,
    TEST_LOSS,
    TRAIN_ACC,
    TRAIN_BATCH_ID,
    TRAIN_LOSS,
    VAL_ACC,
    VAL_LOSS,
)
from specialk.core.utils import batch_texts, check_torch_device, log, namespace_to_dict
from specialk.datasets.dataloaders import (
    init_classification_dataloaders as init_dataloaders,
)
from specialk.models.classifier.onmt.CNNModels import ConvNet
from specialk.models.tokenizer import Vocabulary

warnings.filterwarnings("ignore", message="A parameter name that contains")


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
        metrics = {VAL_ACC: acc, VAL_LOSS: loss}
        self.log_dict(metrics, batch_size=batch["text"].size(0))
        return metrics

    def test_step(self, batch: dict, batch_idx: int):
        loss, acc = self._shared_eval_step(batch, batch_idx)
        metrics = {TEST_ACC: acc, TEST_LOSS: loss}
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
            {TRAIN_ACC: accuracy, TRAIN_BATCH_ID: batch_idx, TRAIN_LOSS: loss},
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
        metrics = {VAL_ACC: acc, VAL_LOSS: loss}
        self.log_dict(metrics, batch_size=batch["text"].size(0))
        return metrics

    def test_step(self, batch: dict, batch_idx: int):
        loss, acc = self._shared_eval_step(batch, batch_idx)
        metrics = {TEST_ACC: acc, TEST_LOSS: loss}
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


class BERTClassifier(TextClassifier):
    def __init__(
        self,
        name: str,
        model_base_name: str = "distilbert/distilbert-base-cased",
        peft_config: Optional[LoraConfig] = None,
        vocabulary_size: Optional[int] = None,
        sequence_length: Optional[int] = None,
        tokenizer: Optional[Vocabulary] = None,
    ):
        """Uses Huggingface based BERT models."""
        if tokenizer:
            log.warn(
                "You don't need to set a tokenizer, we're using HuggingFace's pre-trained models."
            )
        if vocabulary_size:
            log.warn(
                "You don't need to set a vocabulary size, we're using HuggingFace's pre-trained models."
            )

        super().__init__(
            name=name,
            vocabulary_size=vocabulary_size,
            sequence_length=sequence_length,
            tokenizer=tokenizer,
        )
        self.model_base_name = model_base_name
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_base_name)
        self._load_base_model()

        self.peft_config = None
        if peft_config:
            assert peft_config.task_type == TaskType.SEQ_CLS
            self.model = get_peft_model(self.base_model, peft_config, adapter_name=name)
            self.peft_config = peft_config
            log.info("Using PEFT.", config=self.peft_config)
        else:
            self.model = self.base_model
        self.save_hyperparameters(logger=False)

    def _load_base_model(self):
        """Load the base model from HuggingFace."""
        log.info("Loading base model")
        self.base_model = AutoModelForSequenceClassification.from_pretrained(
            self.model_base_name
        )
        config = AutoConfig.from_pretrained(self.model_base_name)
        log.info("Successfully loaded model and config.")
        self.sequence_length = config.max_position_embeddings

    def training_step(self, batch: dict, batch_idx: int) -> torch.Tensor:
        """Run Training step.

        Args:
            batch (dict): individual batch generated from the DataLoader.
            batch_idx (int): ID corresponding to the batch.

        Returns:
            torch.Tensor: Returns loss.
        """
        x, y = batch["text"], batch["label"]

        if x.dim() == 3:
            x = x.squeeze(2, 1)

        batch_size, _ = x.size()

        x_mask = x != self.tokenizer.pad_token_id
        y_hat = self.model(x, attention_mask=x_mask, labels=y)

        loss: torch.Tensor = y_hat.loss
        accuracy = self.calculate_classification_metrics(y_hat.logits, y)

        self.log_dict(
            {TRAIN_ACC: accuracy, TRAIN_BATCH_ID: batch_idx, TRAIN_LOSS: loss},
            batch_size=batch_size,
        )
        return loss

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

        if x.dim() == 3:
            x = x.squeeze(2, 1)

        x_mask = x != self.tokenizer.pad_token_id
        y_hat = self.model(x, attention_mask=x_mask, labels=y)

        loss: torch.Tensor = y_hat.loss
        accuracy = self.calculate_classification_metrics(y_hat.logits, y)
        return loss, accuracy

    def validation_step(self, batch: dict, batch_idx: int):
        loss, acc = self._shared_eval_step(batch, batch_idx)
        metrics = {VAL_ACC: acc, VAL_LOSS: loss}
        self.log_dict(metrics, batch_size=batch["text"].size(0))
        return metrics

    def test_step(self, batch: dict, batch_idx: int):
        loss, acc = self._shared_eval_step(batch, batch_idx)
        metrics = {TEST_ACC: acc, TEST_LOSS: loss}
        self.log_dict(metrics)
        return metrics

    def load_peft_from_checkpoint(self, checkpoint) -> PeftModel:
        """For some reason, the PeftModel.from_pretrained works, when
        directly setting the weights from the state_dict doesn't.

        I've spent way too long trying to debug this."""

        peft_config = checkpoint["hyper_parameters"]["peft_config"]
        peft_state_dict = checkpoint["peft_state_dict"]

        # TODO change this with a fake filesytem
        with tempfile.TemporaryDirectory() as temp_dir:
            # Save the modified config to the temp directory
            peft_config.save_pretrained(temp_dir)

            # Save the filtered state dict to the temp directory
            path_peft_tensors = Path(temp_dir) / "adapter_model.safetensors"
            safetensors.torch.save_file(peft_state_dict, path_peft_tensors)

            # Use the standard PEFT loading mechanism.
            peft_model = PeftModel.from_pretrained(
                self.base_model,
                temp_dir,
                adapter_name=self.name,
            )

        return peft_model

    def on_save_checkpoint(self, checkpoint: dict):
        if self.peft_config:
            # Save only the PEFT state dict
            peft_state_dict = get_peft_model_state_dict(
                self.model, adapter_name=self.name
            )
            checkpoint["peft_state_dict"] = peft_state_dict
            # you don't need it because we're only saving peft_state_dict.
            del checkpoint["state_dict"]
        else:
            # For non-PEFT models, save the entire state dict
            checkpoint["state_dict"] = self.model.state_dict()

    def on_load_checkpoint(self, checkpoint):
        # Reload the base model to ensure clean slate
        if "peft_state_dict" in checkpoint:
            # Re-initialize the PEFT model
            self.model = self.load_peft_from_checkpoint(checkpoint)

        else:
            # For non-PEFT models, load the entire state dict
            self.model.load_state_dict(checkpoint["state_dict"])

    @classmethod
    def load_from_checkpoint(cls, checkpoint_path, map_location=None):
        checkpoint = torch.load(checkpoint_path, map_location=map_location)

        model = cls(
            name=checkpoint["hyper_parameters"]["name"],
            model_base_name=checkpoint["hyper_parameters"]["model_base_name"],
            peft_config=checkpoint["hyper_parameters"].get("peft_config"),
        )

        model.on_load_checkpoint(checkpoint)
        return model

    def configure_optimizers(self):
        # Implement your optimizer configuration here
        return torch.optim.AdamW(self.model.parameters(), lr=2e-5)

    def calculate_classification_metrics(
        self, logits: Float[Tensor, "batch 2"], labels: Int[Tensor, "batch"]
    ):
        """
        Calculate classification metrics.

        HuggingFace BERTs will have two outputs instead of one,
        (one for each binary class). This is for flexibility reasons.

        (Provides confidence in both directions). Note that this isn't used
        as a loss metric, so we don't need to keep track of gradients here.
        """
        preds = (
            logits[:, 1] > logits[:, 0]
        ).float()  # Positive class if logits[1] > logits[0]
        return (preds == labels).float().mean()
