from pathlib import Path

import pandas as pd
import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchmetrics.functional import accuracy

from datasets import Dataset, load_dataset
from datasets.exceptions import DatasetGenerationError
from specialk.core.constants import PROJECT_DIR, SOURCE, TARGET
from specialk.core.utils import log
from specialk.models.classifier.onmt.CNNModels import ConvNet
from specialk.models.tokenizer import (
    BPEVocabulary,
    HuggingFaceVocabulary,
    WordVocabulary,
)
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
def hf_bert_tokenizer() -> HuggingFaceVocabulary:
    return HuggingFaceVocabulary(
        name="bert-base-uncased",
        pretrained_model_name_or_path="bert-base-uncased",
        max_length=512,
    )


@pytest.fixture(scope="session")
def hf_distilbert_tokenizer() -> HuggingFaceVocabulary:
    model_name = "distilbert/distilbert-base-cased"
    return HuggingFaceVocabulary(
        name=model_name,
        pretrained_model_name_or_path=model_name,
        max_length=512,
    )


@pytest.fixture(scope="session", autouse=True)
def mt_dataset() -> Dataset:
    dataset_path = PROJECT_DIR / "tests/test_files/datasets/en_fr.parquet"
    return Dataset.from_pandas(pd.read_parquet(dataset_path)[:10])


@pytest.fixture(scope="session")
def hf_marianmt_tokenizer() -> HuggingFaceVocabulary:
    model_name = "Helsinki-NLP/opus-mt-fr-en"
    return HuggingFaceVocabulary(
        name=model_name,
        pretrained_model_name_or_path=model_name,
        max_length=100,
    )


@pytest.fixture(scope="session")
def hf_marianmt_dataloader(
    mt_dataset: Dataset, hf_marianmt_tokenizer: HuggingFaceVocabulary
):
    def tokenize(example):
        # perform tokenization at this stage.
        example[SOURCE] = hf_marianmt_tokenizer.to_tensor(example[SOURCE]).squeeze(0)
        example[TARGET] = hf_marianmt_tokenizer.to_tensor(example[TARGET]).squeeze(0)
        return example

    tokenized_dataset = mt_dataset.with_format("torch").map(tokenize)
    dataloader = DataLoader(tokenized_dataset, batch_size=BATCH_SIZE, shuffle=True)
    return dataloader


@pytest.fixture(scope="session")
def hf_bert_dataloader(dataset: Dataset, hf_bert_tokenizer: HuggingFaceVocabulary):
    def tokenize(example):
        # perform tokenization at this stage.
        example["text"] = hf_bert_tokenizer.to_tensor(example["text"])
        return example

    tokenized_dataset = dataset.with_format("torch").map(tokenize)
    dataloader = DataLoader(tokenized_dataset, batch_size=BATCH_SIZE, shuffle=True)
    return dataloader


@pytest.fixture(scope="session")
def hf_distilbert_dataloader(
    dataset: Dataset, hf_distilbert_tokenizer: HuggingFaceVocabulary
):
    def tokenize(example):
        # perform tokenization at this stage.
        example["text"] = hf_distilbert_tokenizer.to_tensor(example["text"])
        return example

    tokenized_dataset = dataset.with_format("torch").map(tokenize)
    dataloader = DataLoader(tokenized_dataset, batch_size=BATCH_SIZE, shuffle=True)
    return dataloader


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
