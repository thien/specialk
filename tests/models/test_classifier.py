from pathlib import Path

import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader

from datasets import Dataset, load_dataset
from datasets.exceptions import DatasetGenerationError
from specialk.core.utils import log
from specialk.classifier.onmt.CNNModels import ConvNet
from specialk.lib.tokenizer import BPEVocabulary, WordVocabulary
from tests.tokenizer.test_tokenizer import PCT_BPE, VOCABULARY_SIZE
from torch.autograd import Variable
from specialk.classifier.trainer import memory_efficient_loss

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
    tokenizer_filepath = Path(dirpath) / "bpe_tokenizer"

    bpe_tokenizer = BPEVocabulary(
        "source",
        filename=tokenizer_filepath,
        vocab_size=VOCABULARY_SIZE,
        max_length=SEQUENCE_LENGTH,
        pct_bpe=PCT_BPE,
    )
    bpe_tokenizer.load()
    return bpe_tokenizer


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

    log.info("model pooling size", maxpool=model.maxpool)
    batch = next(iter(bpe_dataloader))
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

    assert one_hot.shape == (SEQUENCE_LENGTH, BATCH_SIZE, VOCABULARY_SIZE)

    y_hat = model(one_hot)

    y = y.unsqueeze(0).unsqueeze(-1)

    assert y_hat.shape == (BATCH_SIZE, 1)
    assert y.shape == (1, BATCH_SIZE, 1)

    log.info(
        "shapes",
        x=x.shape,
        one_hot=one_hot.shape,
    )
    log.info("shapes", y_hat=y_hat.shape, y=y.shape)

    _, _ = memory_efficient_loss(y_hat, y, nn.BCELoss())
