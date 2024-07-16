from pathlib import Path

import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader
from jaxtyping import Float, Int
from torch import Tensor
from datasets import Dataset
import pandas as pd
from specialk.core.utils import log
import torch.nn.functional as F
from specialk.models.tokenizer import SentencePieceVocabulary
from specialk.core.constants import PROJECT_DIR, PAD
from tests.tokenizer.test_tokenizer import VOCABULARY_SIZE
from specialk.models.mt_model import TransformerModule, RNNModule
from specialk.models.transformer.pytorch_transformer import (
    PyTorchTransformerModule,
    PyTorchTransformerModel,
)

dirpath: Path = Path("tests/tokenizer/test_files")
tokenizer_path: Path = (
    PROJECT_DIR / "assets" / "tokenizer" / "sentencepiece" / "enfr.model"
)
dataset_path = PROJECT_DIR / "tests/test_files/datasets/en_fr.parquet"

BATCH_SIZE = 25
SEQUENCE_LENGTH = 100

torch.manual_seed(1337)


@pytest.fixture(scope="session", autouse=True)
def dataset() -> Dataset:
    return Dataset.from_pandas(pd.read_parquet(dataset_path))


@pytest.fixture(scope="session")
def spm_tokenizer() -> SentencePieceVocabulary:
    return SentencePieceVocabulary.from_file(tokenizer_path, 100)


@pytest.fixture(scope="session")
def bpe_dataloader(dataset: Dataset, spm_tokenizer: SentencePieceVocabulary):
    def tokenize(example):
        # perform tokenization at this stage.
        example["source"] = spm_tokenizer.to_tensor(example["source"])
        example["target"] = spm_tokenizer.to_tensor(example["target"])
        return example

    tokenized_dataset = dataset.with_format("torch").map(tokenize)
    dataloader = DataLoader(tokenized_dataset, batch_size=BATCH_SIZE, shuffle=True)
    return dataloader, spm_tokenizer


def test_transformer_inference(bpe_dataloader):
    dataloader, tokenizer = bpe_dataloader
    model = TransformerModule(
        name="transformer_1",
        vocabulary_size=VOCABULARY_SIZE,
        sequence_length=SEQUENCE_LENGTH,
    )
    model.PAD = 0
    model.eval()
    criterion = nn.BCELoss()

    batch: dict = next(iter(dataloader))
    x: torch.Tensor = batch["source"].squeeze(1)
    y: torch.Tensor = batch["target"].squeeze(1)
    batch_size: int = x.shape[0]

    len_x = (x == PAD).sum(dim=1)
    len_y = (y == PAD).sum(dim=1)

    assert len_x.shape[0] == batch_size

    print(x.shape, len_x.shape, y.shape, len_y.shape)

    m = model.model

    # forward pass
    enc_output, *_ = m.encoder(x, len_x)
    dec_output, *_ = m.decoder(y, len_y, x, enc_output)
    y_hat = m.generator(dec_output) * m.x_logit_scale

    log.info("shapes", y=y.shape, y_hat=y_hat.shape)

    _ = criterion(y_hat, y.float())


def test_pt_transformer_inference(bpe_dataloader):
    dataloader, tokenizer = bpe_dataloader
    model = PyTorchTransformerModule(
        name="transformer_1",
        vocabulary_size=tokenizer.vocab_size,
        sequence_length=SEQUENCE_LENGTH,
        dim_model=32,
        n_heads=2,
        num_encoder_layers=1,
        num_decoder_layers=1,
    )
    model.model: PyTorchTransformerModel
    model.PAD = 0
    model.eval()

    batch: dict = next(iter(dataloader))
    x: torch.Tensor = batch["source"].squeeze(1)
    y: torch.Tensor = batch["target"].squeeze(1)
    batch_size: int = x.shape[0]

    len_x = (x == PAD).sum(dim=1)
    len_y = (y == PAD).sum(dim=1)

    assert len_x.shape[0] == batch_size

    log.info("shapes", x=x.shape, len_x=len_x.shape, y=y.shape, len_y=len_y.shape)

    # forward pass
    # x = model.
    y_hat: Float[Tensor, "batch seq_length vocab"] = model.model.forward(x, y)

    log.info("shapes", y=y.shape, y_hat=y_hat.shape)

    # flatten tensors for loss function.
    y_hat: Float[Tensor, "batch_size*seq_length, vocab"] = y_hat.view(
        -1, y_hat.size(-1)
    )
    y: Int[Tensor, "batch_size*vocab"] = y.view(-1)
    _ = F.cross_entropy(y_hat, y, ignore_index=model.PAD, reduction="sum")
