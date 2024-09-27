from pathlib import Path

import pytest
import torch
from torch.utils.data import DataLoader

from datasets import Dataset, load_dataset
from datasets.exceptions import DatasetGenerationError
from specialk.core.utils import log
from specialk.models.tokenizer import BPEVocabulary, WordVocabulary

dirpath = "tests/tokenizer/test_files"
dev_path = "/Users/t/Projects/datasets/political/political_data/*"

BATCH_SIZE = 2

torch.manual_seed(1337)


@pytest.fixture(scope="module", autouse=True)
def dataset() -> Dataset:
    try:
        dataset = load_dataset("thien/political", split="eval[:5%]")
    except DatasetGenerationError:
        dataset = Dataset.from_parquet(dev_path)
    dataset = dataset.class_encode_column("label")
    return dataset


@pytest.fixture(scope="module")
def bpe_tokenizer() -> BPEVocabulary:
    tokenizer_filepath = Path(dirpath) / "bpe_tokenizer"
    tokenizer = BPEVocabulary.from_file(tokenizer_filepath)
    return tokenizer


@pytest.fixture(scope="module")
def word_tokenizer() -> WordVocabulary:
    tokenizer_filepath = Path(dirpath) / "word_tokenizer"
    word_tokenizer = WordVocabulary.from_file(tokenizer_filepath)
    return word_tokenizer


@pytest.fixture(scope="module")
def bpe_dataloader(dataset: Dataset, bpe_tokenizer: BPEVocabulary):
    def tokenize(example):
        # perform tokenization at this stage.
        example["text"] = bpe_tokenizer.to_tensor(example["text"])
        return example

    tokenized_dataset = dataset.with_format("torch").map(tokenize)
    dataloader = DataLoader(tokenized_dataset, batch_size=BATCH_SIZE, shuffle=True)
    return dataloader


@pytest.mark.lightweight
def test_make_dataloader(dataset):
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE)
    batch = next(iter(dataloader))
    assert isinstance(batch["text"][0], str)


@pytest.mark.heavyweight
def test_dataloader_tokenized_bpe(dataset: Dataset, bpe_tokenizer: BPEVocabulary):
    def tokenize(example):
        # perform tokenization at this stage.
        example["text"] = bpe_tokenizer.to_tensor(example["text"])
        return example

    tokenized_dataset = dataset.with_format("torch").map(tokenize)
    dataloader = DataLoader(tokenized_dataset, batch_size=BATCH_SIZE, shuffle=False)
    batch = next(iter(dataloader))
    # number chosen by the seed.
    batch_idx = batch["id"]
    expected_idx = torch.Tensor([0, 1])
    assert torch.all(batch_idx.eq(expected_idx))
    assert isinstance(batch["text"], torch.Tensor)
    batch["text"] = batch["text"].reshape(2, -1)
    assert batch["text"].shape == torch.Size([BATCH_SIZE, bpe_tokenizer.max_length])
    log.debug("batch shape", shape=batch["text"].shape)
    assert isinstance(batch["text"][0][0].item(), int)


@pytest.mark.heavyweight
def test_dataloader_tokenized_word(dataset: Dataset, word_tokenizer: WordVocabulary):
    def tokenize(example):
        # perform tokenization at this stage.
        example["text"] = word_tokenizer.to_tensor(example["text"])
        return example

    tokenized_dataset = dataset.with_format("torch").map(tokenize)
    dataloader = DataLoader(tokenized_dataset, batch_size=BATCH_SIZE, shuffle=False)
    batch = next(iter(dataloader))
    # number chosen by the seed.
    batch_idx = batch["id"]
    expected_idx = torch.Tensor([0, 1])
    assert torch.all(batch_idx.eq(expected_idx))
    assert isinstance(batch["text"], torch.Tensor)
    batch["text"] = batch["text"].reshape(2, -1)
    assert batch["text"].shape == torch.Size([BATCH_SIZE, word_tokenizer.max_length])
    log.debug("batch shape", shape=batch["text"].shape)
    assert isinstance(batch["text"][0][0].item(), int)


@pytest.mark.heavyweight
def test_dataloader_class_label(dataset: Dataset):
    torch_dataset = dataset.with_format("torch")
    log.info("features", features=dataset.features)
    assert isinstance(torch_dataset["label"][0].item(), int)
