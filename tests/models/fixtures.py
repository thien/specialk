from pathlib import Path

import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchmetrics.functional import accuracy

from datasets import Dataset, load_dataset
from datasets.exceptions import DatasetGenerationError
from specialk.core.utils import log
from specialk.models.classifier.onmt.CNNModels import ConvNet
from specialk.models.tokenizer import BPEVocabulary, WordVocabulary
from tests.tokenizer.test_tokenizer import VOCABULARY_SIZE

dirpath = "tests/tokenizer/test_files"
dev_path = (
    "/Users/t/Projects/datasets/political/political_data/democratic_only.dev.en.parquet"
)

BATCH_SIZE = 128
SEQUENCE_LENGTH = 100

torch.manual_seed(1337)

@pytest.fixture(scope="session", autouse=True)
def dataset() -> Dataset:
    try:
        dataset = load_dataset("thien/political", split="eval")
    except DatasetGenerationError:
        dataset = Dataset.from_parquet(dev_path)
    dataset = dataset.class_encode_column("label")
    return dataset


@pytest.fixture(scope="session")
def bpe_tokenizer() -> BPEVocabulary:
    return BPEVocabulary.from_file(Path(dirpath) / "bpe_tokenizer")


@pytest.fixture(scope="session")
def word_tokenizer() -> WordVocabulary:
    return WordVocabulary.from_file(Path(dirpath) / "word_tokenizer")


@pytest.fixture(scope="session")
def word_dataloader(dataset: Dataset, word_tokenizer: WordVocabulary):
    def tokenize(example):
        # perform tokenization at this stage.
        example["text"] = word_tokenizer.to_tensor(example["text"])
        return example

    tokenized_dataset = dataset.with_format("torch").map(tokenize)
    dataloader = DataLoader(tokenized_dataset, batch_size=BATCH_SIZE, shuffle=True)
    return dataloader


@pytest.fixture(scope="session")
def bpe_dataloader(dataset: Dataset, bpe_tokenizer: BPEVocabulary):
    def tokenize(example):
        # perform tokenization at this stage.
        example["text"] = bpe_tokenizer.to_tensor(example["text"])
        return example

    tokenized_dataset = dataset.with_format("torch").map(tokenize)
    dataloader = DataLoader(tokenized_dataset, batch_size=BATCH_SIZE, shuffle=True)
    return dataloader
