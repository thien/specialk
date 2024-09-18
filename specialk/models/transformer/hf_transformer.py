"""
Huggingface Transformers.

This is largely a wrapper that combines NMTModule, Peft and Huggingface Transformers.
This is to leverage existing pre-trained modules.
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Callable, List, Optional, Tuple, Union

import safetensors
import torch
from peft import (
    LoraConfig,
    PeftModel,
    TaskType,
    get_peft_model,
    get_peft_model_state_dict,
)
from torch.nn import LayerNorm
from torch.optim import AdamW
from transformers import AutoConfig, AutoModelForSeq2SeqLM, AutoTokenizer

from specialk.core.constants import TRAIN_ACC, TRAIN_BATCH_ID, TRAIN_LOSS
from specialk.core.utils import log
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
