"""Checkpoint helpers shared by the HuggingFace-backed Lightning modules.

`BERTClassifier` and `MarianMTModule` support optional LoRA adapters.  A PEFT
checkpoint stores only the adapter weights (plus any ``modules_to_save``
weights), not the frozen base model.  Restoring them with PEFT's state-dict
helper keeps everything in memory and handles adapter-name remapping, so no
temporary adapter directory is needed.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import safetensors.torch
import torch
from peft import PeftConfig, load_peft_weights, set_peft_model_state_dict
from peft.utils.save_and_load import get_peft_model_state_dict


def save_checkpoint(module: Any, checkpoint: dict) -> None:
    """Put the weights that need persisting onto a Lightning checkpoint."""
    if module.peft_config:
        checkpoint["peft_state_dict"] = get_peft_model_state_dict(
            module.model, adapter_name=module.name
        )
        # The full state dict is redundant: the base model is frozen and its
        # pretrained weights are reloaded by the constructor.
        checkpoint.pop("state_dict", None)
    else:
        checkpoint["state_dict"] = module.model.state_dict()


def load_checkpoint(module: Any, checkpoint: dict) -> None:
    """Restore weights written by :func:`save_checkpoint`."""
    if "peft_state_dict" in checkpoint:
        _load_adapter_state(module, checkpoint["peft_state_dict"])
    else:
        module.model.load_state_dict(checkpoint["state_dict"])


def _load_adapter_state(module: Any, state_dict: dict) -> None:
    result = set_peft_model_state_dict(
        module.model,
        state_dict,
        adapter_name=module.name,
    )
    if result.unexpected_keys:
        raise ValueError(
            f"Adapter weights do not match the model: {result.unexpected_keys}"
        )


def save_adapter(module: Any, save_directory) -> None:
    """Write a standard PEFT adapter directory for ``module``."""
    if not module.peft_config:
        raise ValueError("Cannot save an adapter from a non-PEFT model.")
    save_directory = Path(save_directory)
    save_directory.mkdir(parents=True, exist_ok=True)
    # shortcut: one adapter per module; use PeftModel.save_pretrained for multi-adapter export.
    module.peft_config.save_pretrained(save_directory)
    safetensors.torch.save_file(
        get_peft_model_state_dict(module.model, adapter_name=module.name),
        save_directory / "adapter_model.safetensors",
    )


def load_adapter(
    cls: type,
    adapter_path,
    name: str,
    model_base_name: str | None = None,
    **kwargs,
):
    """Build ``cls`` from a saved adapter directory and restore the adapter."""
    config = PeftConfig.from_pretrained(adapter_path)
    module = cls(
        name=name,
        model_base_name=model_base_name or config.base_model_name_or_path,
        peft_config=config,
        **kwargs,
    )
    _load_adapter_state(module, load_peft_weights(adapter_path, device="cpu"))
    return module


def load_from_checkpoint(
    cls: type,
    checkpoint_path,
    map_location=None,
    weights_only: bool = False,
):
    """Build ``cls`` from the checkpoint hyperparameters and restore its weights.

    ``weights_only=False`` is intentional.  Checkpoints contain PEFT config
    objects, which the secure default introduced in torch 2.6 rejects.  These
    are trusted local checkpoint files, so keep the pre-2.6 behaviour.
    """
    checkpoint = torch.load(
        checkpoint_path, map_location=map_location, weights_only=weights_only
    )
    hyper_parameters = checkpoint["hyper_parameters"]
    model = cls(
        name=hyper_parameters["name"],
        model_base_name=hyper_parameters["model_base_name"],
        peft_config=hyper_parameters.get("peft_config"),
    )
    load_checkpoint(model, checkpoint)
    return model
