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


def test_model_inference(bpe_dataloader):
    model = ConvNet(
        vocab_size=VOCABULARY_SIZE,
        sequence_length=SEQUENCE_LENGTH,
    )
    model.eval()
    criterion = nn.BCELoss()

    batch: dict = next(iter(bpe_dataloader))
    x: torch.Tensor
    y: torch.Tensor
    x, y = batch["text"], batch["label"]

    x = x.squeeze(2, 1)
    seq_len: int = x.size(1)
    batch_size: int = x.size(0)
    vocab_size: int = model.vocab_size

    assert x.shape == (BATCH_SIZE, SEQUENCE_LENGTH)
    assert vocab_size == VOCABULARY_SIZE
    assert batch_size == BATCH_SIZE
    assert (
        batch_size >= 50
    ), f"Batch size needs to be greater than 50 (currently {batch_size})."

    one_hot = torch.zeros(seq_len, batch_size, vocab_size).scatter_(
        2, torch.unsqueeze(x.T, 2), 1
    )

    # note that this is a lot slower than the implementation above.
    # one_hot = torch.nn.functional.one_hot(x.T, num_classes=VOCABULARY_SIZE).float()

    assert one_hot.shape == (SEQUENCE_LENGTH, BATCH_SIZE, VOCABULARY_SIZE)

    y_hat = model(one_hot).squeeze(-1)

    assert y_hat.shape[0] == BATCH_SIZE
    assert y.shape[0] == BATCH_SIZE

    log.info("shapes", y=y.shape, y_hat=y_hat.shape)

    _ = criterion(y_hat, y.float())


def test_accuracy():
    y = torch.LongTensor([0, 0, 1, 1, 1, 1]).float()
    y_pred = torch.LongTensor([0, 0, 0, 1, 1, 1])
    assert y.shape == y_pred.shape
    acc = accuracy(y_pred, y, "binary")
    assert acc == pytest.approx(0.833, 0.1)
