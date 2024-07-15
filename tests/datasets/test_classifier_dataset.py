from pathlib import Path

import pytest
import torch
from torch.utils.data import DataLoader

from datasets import Dataset, load_dataset
from datasets.exceptions import DatasetGenerationError
from specialk.core.utils import log
from specialk.models.tokenizer import BPEVocabulary, WordVocabulary
from tests.tokenizer.test_tokenizer import PCT_BPE, SEQUENCE_LENGTH, VOCABULARY_SIZE

dirpath = "tests/tokenizer/test_files"
dev_path = "/Users/t/Projects/datasets/political/political_data/*"

BATCH_SIZE = 2

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
    tokenizer_filepath = Path(dirpath) / "bpe_tokenizer"

    bpe_tokenizer = BPEVocabulary(
        "source",
        filename=tokenizer_filepath,
        vocab_size=VOCABULARY_SIZE,
        max_length=60,
        pct_bpe=PCT_BPE,
    )
    bpe_tokenizer.load()
    return bpe_tokenizer


@pytest.fixture(scope="session")
def word_tokenizer() -> WordVocabulary:
    tokenizer_filepath = Path(dirpath) / "word_tokenizer"

    word_tokenizer = WordVocabulary(
        name="source",
        filename=tokenizer_filepath,
        vocab_size=VOCABULARY_SIZE,
        max_length=60,
        lower=True,
    )
    word_tokenizer.load()
    return word_tokenizer


@pytest.fixture(scope="session")
def bpe_dataloader(dataset: Dataset, bpe_tokenizer: BPEVocabulary):
    def tokenize(example):
        # perform tokenization at this stage.
        example["text"] = bpe_tokenizer.to_tensor(example["text"])
        return example

    tokenized_dataset = dataset.with_format("torch").map(tokenize)
    dataloader = DataLoader(tokenized_dataset, batch_size=BATCH_SIZE, shuffle=True)
    return dataloader


def test_make_dataloader(dataset):
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE)
    batch = next(iter(dataloader))
    assert isinstance(batch["text"][0], str)


def test_dataloader_tokenized_bpe(dataset: Dataset, bpe_tokenizer: BPEVocabulary):
    def tokenize(example):
        # perform tokenization at this stage.
        example["text"] = bpe_tokenizer.to_tensor(example["text"])
        return example

    tokenized_dataset = dataset.with_format("torch").map(tokenize)
    dataloader = DataLoader(tokenized_dataset, batch_size=BATCH_SIZE, shuffle=True)
    batch = next(iter(dataloader))
    # number chosen by the seed.
    batch_idx = batch["id"]
    expected_idx = torch.Tensor([860, 1526])
    assert torch.all(batch_idx.eq(expected_idx))
    assert isinstance(batch["text"], torch.Tensor)
    batch["text"] = batch["text"].reshape(2, -1)
    assert batch["text"].shape == torch.Size([BATCH_SIZE, bpe_tokenizer.max_length])
    log.debug("batch shape", shape=batch["text"].shape)
    assert isinstance(batch["text"][0][0].item(), int)


def test_dataloader_tokenized_word(dataset: Dataset, word_tokenizer: WordVocabulary):
    def tokenize(example):
        # perform tokenization at this stage.
        example["text"] = word_tokenizer.to_tensor(example["text"])
        return example

    tokenized_dataset = dataset.with_format("torch").map(tokenize)
    dataloader = DataLoader(tokenized_dataset, batch_size=BATCH_SIZE, shuffle=True)
    batch = next(iter(dataloader))
    # number chosen by the seed.
    batch_idx = batch["id"]
    expected_idx = torch.Tensor([757, 1988])
    assert torch.all(batch_idx.eq(expected_idx))
    assert isinstance(batch["text"], torch.Tensor)
    batch["text"] = batch["text"].reshape(2, -1)
    assert batch["text"].shape == torch.Size([BATCH_SIZE, word_tokenizer.max_length])
    log.debug("batch shape", shape=batch["text"].shape)
    assert isinstance(batch["text"][0][0].item(), int)


def test_dataloader_class_label(dataset: Dataset):
    torch_dataset = dataset.with_format("torch")
    log.info("features", features=dataset.features)
    assert isinstance(torch_dataset["label"][0].item(), int)
