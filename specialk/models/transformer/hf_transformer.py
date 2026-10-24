"""
Huggingface Transformers.

This is largely a wrapper that combines NMTModule, Peft and Huggingface Transformers.
This is to leverage existing pre-trained modules.
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Tuple

import torch
from peft import get_peft_model
from peft.tuners.lora.config import LoraConfig
from peft.utils.peft_types import TaskType
from torch import Tensor
from transformers import AutoConfig, AutoModelForSeq2SeqLM, AutoTokenizer

from specialk.core.constants import TRAIN_ACC, TRAIN_BATCH_ID, TRAIN_LOSS
from specialk.core.utils import log
from specialk.models import peft_utils
from specialk.models.mt_model import NMTModule
from specialk.models.tokenizer import Vocabulary


class MarianMTModule(NMTModule):
    def __init__(
        self,
        name: str,
        model_base_name: str = "Helsinki-NLP/opus-mt-fr-en",
        peft_config: Optional[LoraConfig] = None,
        tokenizer: Optional[Vocabulary] = None,
    ):
        """Uses Huggingface based pre-trained models."""

        super().__init__(
            name=name,
            vocabulary_size=1,
            sequence_length=512,
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

    def forward(self, x: Tensor, y: Tensor) -> Tensor:
        x_mask = x != self.tokenizer.pad_token_id
        return self.model(x, labels=y, attention_mask=x_mask).logits

    def get_lora_config_modules(
        self, num_decoder_layers=6, add_encoder=False
    ) -> List[str]:
        if add_encoder:
            raise NotImplementedError
        modules = []
        num_decoder_layers = (
            len(self.model.model.decoder.layers)
            if num_decoder_layers is None
            else num_decoder_layers
        )
        for layer in range(num_decoder_layers):
            for proj in ["q_proj", "k_proj", "v_proj"]:
                modules.append(f"decoder.layers.{layer}.encoder_attn.{proj}")
        modules.append("lm_head")
        return modules

    def _load_base_model(self):
        """Load the base model from HuggingFace."""
        log.info("Loading base model")
        self.base_model = AutoModelForSeq2SeqLM.from_pretrained(self.model_base_name)
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
        x, y = batch["source"], batch["target"]
        batch_size = x.shape[0]

        x_mask = x != self.tokenizer.pad_token_id
        y_hat = self.model(x, labels=y, attention_mask=x_mask)

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
        x, y = batch["source"], batch["target"]

        x_mask = x != self.tokenizer.pad_token_id
        y_hat = self.model(x, labels=y, attention_mask=x_mask)

        loss: torch.Tensor = y_hat.loss
        accuracy = self.calculate_classification_metrics(y_hat.logits, y)
        return loss, accuracy

    def on_save_checkpoint(self, checkpoint: dict):
        peft_utils.save_checkpoint(self, checkpoint)

    def on_load_checkpoint(self, checkpoint: dict):
        peft_utils.load_checkpoint(self, checkpoint)

    def _freeze_non_lora_weights(self):
        for name, param in self.model.named_parameters():
            if "lora_" not in name:
                param.requires_grad = False

    @classmethod
    def load_from_checkpoint(
        cls, checkpoint_path: Path, map_location=None, weights_only: bool = False
    ):
        return peft_utils.load_from_checkpoint(
            cls, checkpoint_path, map_location, weights_only
        )

    def save_adapter(self, save_directory):
        peft_utils.save_adapter(self, save_directory)

    @classmethod
    def from_adapter(
        cls,
        adapter_path,
        name: str,
        model_base_name: Optional[str] = None,
        **kwargs,
    ):
        return peft_utils.load_adapter(
            cls, adapter_path, name, model_base_name, **kwargs
        )
