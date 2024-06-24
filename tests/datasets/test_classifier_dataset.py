from datasets import Dataset, load_dataset
from torch.utils.data import DataLoader
import pytest
import torch
from specialk.lib.tokenizer import WordVocabulary, BPEVocabulary
from tests.tokenizer.test_tokenizer import VOCABULARY_SIZE, SEQUENCE_LENGTH, PCT_BPE
from pathlib import Path
from specialk.core.utils import log

dirpath = "tests/tokenizer/test_files"

BATCH_SIZE = 2

torch.manual_seed(1337)


@pytest.fixture(scope="session", autouse=True)
def dataset() -> Dataset:
    return load_dataset("thien/political", split="eval")


@pytest.fixture()
def bpe_tokenizer() -> BPEVocabulary:
    tokenizer_filepath = Path(dirpath) / "bpe_tokenizer"

    bpe_tokenizer = BPEVocabulary(
        "source", tokenizer_filepath, VOCABULARY_SIZE, 60, PCT_BPE
    )
    bpe_tokenizer.load()
    return bpe_tokenizer


@pytest.fixture()
def word_tokenizer() -> WordVocabulary:
    tokenizer_filepath = Path(dirpath) / "word_tokenizer"

    word_tokenizer = WordVocabulary(
        "source", tokenizer_filepath, VOCABULARY_SIZE, 60, True
    )
    word_tokenizer.load()
    return word_tokenizer


def test_make_dataloader(dataset):
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE)
    batch = next(iter(dataloader))
    assert isinstance(batch["text"][0], str)


def test_dataloader_tokenized_bpe(dataset: Dataset, bpe_tokenizer: BPEVocabulary):
    def tokenize(example):
        example["text"] = bpe_tokenizer.to_tensor(example["text"])
        return example

    tokenized_dataset = dataset.with_format("torch").map(tokenize)
    dataloader = DataLoader(tokenized_dataset, batch_size=BATCH_SIZE, shuffle=True)
    batch = next(iter(dataloader))
    # number chosen by the seed.
    batch_idx = batch["__index_level_0__"]
    expected_idx = torch.Tensor([82860, 83526])
    assert torch.all(batch_idx.eq(expected_idx))
    log.info("batch", batch=batch)
    assert isinstance(batch["text"], torch.Tensor)
    batch["text"] = batch["text"].reshape(
        2, -1
    )  # TODO: this is caused by BPE I think. Should investigate.
    assert batch["text"].shape == torch.Size([BATCH_SIZE, bpe_tokenizer.max_length])
    log.debug("batch shape", shape=batch["text"].shape)
    assert isinstance(batch["text"][0][0].item(), int)


def test_dataloader_tokenized_word(dataset: Dataset, word_tokenizer: WordVocabulary):
    def tokenize(example):
        example["text"] = word_tokenizer.to_tensor(example["text"])
        return example

    tokenized_dataset = dataset.with_format("torch").map(tokenize)
    dataloader = DataLoader(tokenized_dataset, batch_size=BATCH_SIZE, shuffle=True)
    batch = next(iter(dataloader))
    # number chosen by the seed.
    batch_idx = batch["__index_level_0__"]
    expected_idx = torch.Tensor([82757, 81988])
    assert torch.all(batch_idx.eq(expected_idx))
    log.info("batch", batch=batch)
    assert isinstance(batch["text"], torch.Tensor)
    batch["text"] = batch["text"].reshape(
        2, -1
    )  # TODO: this is caused by BPE I think. Should investigate.
    assert batch["text"].shape == torch.Size([BATCH_SIZE, word_tokenizer.max_length])
    log.debug("batch shape", shape=batch["text"].shape)
    assert isinstance(batch["text"][0][0].item(), int)
