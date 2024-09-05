"""
CNN Training Runner.

Trains classifier used for style transfer.
Modified version of the following code:
https://github.com/shrimai/Style-Transfer-Through-Back-Translation/blob/master/classifier/cnn_train.py.


TODO: Check this works on both BPE and word-separated tokenized datasets from the huggingface datasets.
"""

from __future__ import division

import argparse
from pathlib import Path
from typing import List, Optional, Tuple, Union

import lightning.pytorch as pl
import torch
import torch.nn as nn
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.profilers import AdvancedProfiler
from peft import LoraConfig, TaskType
from torch.autograd import Variable
from torch.nn.modules.loss import _Loss as Loss
from torch.utils.data import DataLoader
from tqdm import tqdm

import specialk.models.classifier.onmt as onmt
from datasets import Dataset, load_dataset
from specialk.core.constants import LOGGING_DIR, LOGGING_PERF_NAME, PROJECT_DIR
from specialk.core.utils import check_torch_device, log, namespace_to_dict
from specialk.datasets.dataloaders import (
    init_classification_dataloaders as init_dataloaders,
)
from specialk.models.classifier.models import BERTClassifier, CNNClassifier
from specialk.models.classifier.onmt.CNNModels import ConvNet
from specialk.models.tokenizer import (
    BPEVocabulary,
    HuggingFaceVocabulary,
    SentencePieceVocabulary,
    Vocabulary,
    WordVocabulary,
)

DEVICE: str = check_torch_device()


def init_dataloader(
    dataset: Dataset,
    tokenizer: Vocabulary,
    batch_size: int,
    shuffle=False,
    n_workers: int = 8,
    persistent_workers: bool = True,
    cache_path: Optional[Union[Path, str]] = None,
) -> DataLoader:
    def tokenize(example):
        # perform tokenization at this stage.
        example["text"] = tokenizer.to_tensor(example["text"])
        return example

    if cache_path:
        if isinstance(cache_path, Path):
            cache_path = str(cache_path)
        if not cache_path.lower().endswith(".parquet"):
            cache_path = f"{cache_path}.parquet"

    tokenized_dataset = dataset.with_format("torch").map(
        tokenize, batched=True, num_proc=n_workers, cache_file_name=cache_path
    )
    dataloader = DataLoader(
        tokenized_dataset,
        batch_size=batch_size,
        persistent_workers=persistent_workers,
        shuffle=shuffle,
        num_workers=n_workers,
    )

    return dataloader


def main_new():
    BATCH_SIZE = 128 if DEVICE == "mps" else 680

    # tokenizer
    tokenizer_option = "spm"  # "word"
    # interestingly the bpe tokenizer is a lot slwoer to run instead of the word tokenizer.
    # we should see how well this performs with sentencepience.
    dir_tokenizer = PROJECT_DIR / "assets" / "tokenizer"
    if tokenizer_option == "bpe":
        tokenizer_filepath = dir_tokenizer / "fr_en_bpe"
        tokenizer = BPEVocabulary.from_file(tokenizer_filepath)
        tokenizer.vocab._progress_bar = iter
    elif tokenizer_option == "word":
        # word option.
        tokenizer_filepath = dir_tokenizer / "fr_en_word_moses"
        tokenizer = WordVocabulary.from_file(tokenizer_filepath)
    else:
        # sentencepiece
        tokenizer_filepath = str(dir_tokenizer / "sentencepiece" / "enfr.model")
        tokenizer = SentencePieceVocabulary.from_file(tokenizer_filepath, max_length=72)

    log.info("Loaded tokenizer", tokenizer=tokenizer)

    hf_dataset_name = "thien/political"

    dataset: Dataset = load_dataset(hf_dataset_name)
    dataset = dataset.class_encode_column("label")

    train_dataloader = init_dataloader(
        dataset["train"], tokenizer, BATCH_SIZE, shuffle=True
    )
    val_dataloader = init_dataloader(
        dataset["eval"], tokenizer, BATCH_SIZE, shuffle=False
    )

    task = CNNClassifier(
        "political",
        vocabulary_size=tokenizer.vocab_size,
        sequence_length=tokenizer.max_length,
        tokenizer=tokenizer,
    )

    logger = TensorBoardLogger(LOGGING_DIR, name="pol_classifier")
    logger.log_hyperparams(
        params={
            "batch_size": BATCH_SIZE,
            "tokenizer": tokenizer.__class__.__name__,
            "dataset": hf_dataset_name,
            "tokenizer_path": tokenizer_filepath,
        }
    )

    profiler = AdvancedProfiler(dirpath=logger.log_dir, filename=LOGGING_PERF_NAME)
    trainer = pl.Trainer(
        accelerator=DEVICE,
        max_epochs=2,
        log_every_n_steps=20,
        logger=logger,
        profiler=profiler,
    )

    trainer.fit(
        task, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader
    )


def main_distilbert_peft():
    BATCH_SIZE = 32 if DEVICE == "mps" else 32
    peft_config = LoraConfig(
        task_type=TaskType.SEQ_CLS,
        r=8,
        lora_alpha=32,
        lora_dropout=0.1,
        bias="none",
        target_modules=["q_lin", "k_lin", "v_lin", "out_lin"],
    )

    model_base_name = "distilbert/distilbert-base-cased"
    task = BERTClassifier(
        name="test_distilbert",
        model_base_name=model_base_name,
        peft_config=peft_config,
    )
    tokenizer = HuggingFaceVocabulary(
        name=model_base_name,
        pretrained_model_name_or_path=model_base_name,
        max_length=512,
    )

    hf_dataset_name = "thien/political"
    hf_dataset_name = "thien/publications"

    dataset: Dataset = load_dataset(hf_dataset_name)
    dataset = dataset.class_encode_column("label")

    train_dataloader = init_dataloader(
        dataset["train"], tokenizer, BATCH_SIZE, shuffle=True
    )
    val_dataloader = init_dataloader(
        dataset["eval"], tokenizer, BATCH_SIZE, shuffle=False
    )

    project_name = f"{hf_dataset_name}/{task.name}"

    logger = TensorBoardLogger(LOGGING_DIR, name=project_name)
    logger.log_hyperparams(
        params={
            "batch_size": BATCH_SIZE,
            "tokenizer": tokenizer.__class__.__name__,
            "dataset": hf_dataset_name,
            "model_base_name": model_base_name,
            "tokenizer_path": None,
        }
    )
    REVIEW_RATE = len(train_dataloader) // 32  # for debugging purposes.

    profiler = AdvancedProfiler(dirpath=logger.log_dir, filename=LOGGING_PERF_NAME)
    trainer = pl.Trainer(
        accelerator=DEVICE,
        max_epochs=2,
        log_every_n_steps=20,
        logger=logger,
        profiler=profiler,
        precision="16-mixed",
        val_check_interval=REVIEW_RATE,
        gradient_clip_val=0.5,
        gradient_clip_algorithm="norm",
    )

    trainer.fit(
        task, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader
    )


if __name__ == "__main__":
    # main_new()
    main_distilbert_peft()
