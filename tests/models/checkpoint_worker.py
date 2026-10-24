"""Subprocess worker for the checkpoint round-trip tests.

The save and load phases must run in different interpreters: a same-process
test can hide state that never made it into the checkpoint.  Run this file
directly; it is not a pytest module.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
import torch
from lightning.pytorch import Trainer
from peft import LoraConfig, get_peft_model_state_dict
from peft.utils.peft_types import TaskType
from torch.utils.data import DataLoader

from datasets import Dataset
from specialk.core.constants import SOURCE, TARGET
from specialk.models.classifier.models import BERTClassifier
from specialk.models.tokenizer import HuggingFaceVocabulary
from specialk.models.transformer.hf_transformer import MarianMTModule

ROOT = Path(__file__).resolve().parents[2]
MARIAN_DATA = ROOT / "tests/test_files/datasets/en_fr.parquet"
BERT_BASE = "bert-base-uncased"
MARIAN_BASE = "Helsinki-NLP/opus-mt-fr-en"

BERT_ROWS = [
    ("This is a great movie", 1),
    ("This is a terrible movie", 0),
    ("I loved every minute of it", 1),
    ("I hated every minute of it", 0),
    ("The plot was engaging", 1),
    ("The plot was boring", 0),
    ("A wonderful performance", 1),
    ("A dreadful performance", 0),
]


class _BertDataset(torch.utils.data.Dataset):
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __len__(self):
        return len(BERT_ROWS)

    def __getitem__(self, index):
        text, label = BERT_ROWS[index]
        input_ids = self.tokenizer(
            text,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=32,
        )["input_ids"].squeeze(0)
        return {"text": input_ids, "label": label}


def _build(arch: str, use_peft: bool):
    if arch == "bert":
        peft_config = (
            LoraConfig(
                task_type=TaskType.SEQ_CLS,
                r=8,
                lora_alpha=32,
                lora_dropout=0.0,
                bias="none",
                target_modules=["query", "key", "value"],
            )
            if use_peft
            else None
        )
        return BERTClassifier(
            name="checkpoint-test",
            model_base_name=BERT_BASE,
            peft_config=peft_config,
        )

    peft_config = (
        LoraConfig(
            task_type=TaskType.SEQ_CLS,
            r=8,
            lora_alpha=32,
            lora_dropout=0.0,
            bias="none",
            target_modules=["q_proj", "k_proj", "v_proj"],
        )
        if use_peft
        else None
    )
    return MarianMTModule(
        name="checkpoint-test",
        model_base_name=MARIAN_BASE,
        peft_config=peft_config,
    )


def _dataloader(arch: str, module):
    if arch == "bert":
        return DataLoader(_BertDataset(module.tokenizer), batch_size=8)

    tokenizer = HuggingFaceVocabulary(
        name=MARIAN_BASE,
        pretrained_model_name_or_path=MARIAN_BASE,
        max_length=100,
    )
    dataset = Dataset.from_pandas(pd.read_parquet(MARIAN_DATA)[:8])

    def tokenize(example):
        example[SOURCE] = tokenizer.to_tensor(example[SOURCE]).squeeze(0)
        example[TARGET] = tokenizer.to_tensor(example[TARGET]).squeeze(0)
        return example

    dataset = dataset.with_format("torch").map(tokenize)
    return DataLoader(dataset, batch_size=8, shuffle=False)


def _inputs_and_logits(arch: str, module, batch):
    module.eval()
    if arch == "bert":
        input_ids = batch["text"]
        if input_ids.dim() == 3:
            input_ids = input_ids.squeeze(2, 1)
        inputs = {
            "input_ids": input_ids,
            "attention_mask": input_ids != module.tokenizer.pad_token_id,
        }
        with torch.inference_mode():
            logits = module.model(**inputs).logits
    else:
        source = batch[SOURCE]
        inputs = {
            "input_ids": source,
            "attention_mask": source != module.tokenizer.pad_token_id,
            "labels": batch[TARGET],
        }
        with torch.inference_mode():
            logits = module.model(**inputs).logits
    return inputs, logits


def _weights(module):
    if module.peft_config:
        return get_peft_model_state_dict(module.model, adapter_name=module.name)
    return module.model.state_dict()


def _make_adapter_observable(module):
    """LoRA B is zero-initialised, so make a silent adapter drop visible."""
    if not module.peft_config:
        return
    with torch.no_grad():
        for name, param in module.model.named_parameters():
            if "lora_B" in name:
                param.add_(0.05)
            elif "lora_A" in name:
                param.add_(0.01)


def save(args):
    module = _build(args.arch, args.peft)
    loader = _dataloader(args.arch, module)

    trainer = Trainer(
        max_epochs=1,
        limit_train_batches=1,
        accelerator=args.device,
        devices=1,
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=False,
    )
    trainer.fit(module, train_dataloaders=loader)

    _make_adapter_observable(module)

    batch = next(iter(loader))
    inputs, logits = _inputs_and_logits(args.arch, module, batch)

    trainer.save_checkpoint(args.checkpoint)

    if args.adapter:
        module.save_adapter(args.adapter)

    reference = {
        "peft": bool(module.peft_config),
        "logits": logits.detach().cpu(),
        "inputs": {key: value.detach().cpu() for key, value in inputs.items()},
        "weights": {
            key: value.detach().cpu() for key, value in _weights(module).items()
        },
    }
    torch.save(reference, args.reference)


def _compare(module, reference, args):
    module.to(args.device)
    module.eval()

    inputs = {key: value.to(args.device) for key, value in reference["inputs"].items()}
    with torch.inference_mode():
        logits = module.model(**inputs).logits.cpu()

    weights = {key: value.detach().cpu() for key, value in _weights(module).items()}
    saved_weights = reference["weights"]
    weight_keys_match = set(weights) == set(saved_weights)
    weight_diffs = [
        (weights[key].float() - saved_weights[key].float()).abs().max().item()
        for key in set(weights) & set(saved_weights)
    ]
    weights_max_diff = max(weight_diffs) if weight_diffs else float("inf")

    result = {
        "peft": reference["peft"],
        "device": args.device,
        "n_weights": len(saved_weights),
        "weight_keys_match": weight_keys_match,
        "weights_max_diff": weights_max_diff,
        "logits_max_diff": (logits - reference["logits"]).abs().max().item(),
        "logits_abs_max": reference["logits"].abs().max().item(),
    }
    with open(args.result, "w") as handle:
        json.dump(result, handle)


def load(args):
    reference = torch.load(args.reference, map_location="cpu", weights_only=True)
    module_cls = {"bert": BERTClassifier, "marian": MarianMTModule}[args.arch]
    module = module_cls.load_from_checkpoint(args.checkpoint, map_location="cpu")
    _compare(module, reference, args)


def load_adapter(args):
    reference = torch.load(args.reference, map_location="cpu", weights_only=True)
    module_cls = {"bert": BERTClassifier, "marian": MarianMTModule}[args.arch]
    module = module_cls.from_adapter(args.adapter, name="checkpoint-test")
    _compare(module, reference, args)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("phase", choices=["save", "load", "load-adapter"])
    parser.add_argument("--arch", required=True, choices=["bert", "marian"])
    parser.add_argument("--device", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--reference", required=True)
    parser.add_argument("--result")
    parser.add_argument("--adapter")
    parser.add_argument("--peft", action="store_true")
    args = parser.parse_args()

    if args.phase == "save":
        save(args)
    elif args.phase == "load":
        load(args)
    else:
        load_adapter(args)


if __name__ == "__main__":
    main()
