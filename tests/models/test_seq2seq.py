from pathlib import Path

import pandas as pd
import pytest
import torch
import torch.nn.functional as F
from jaxtyping import Float, Int
from torch import Tensor, nn
from torch.utils.data import DataLoader

from datasets import Dataset
from specialk.core.constants import PAD, PROJECT_DIR, SOURCE, TARGET
from specialk.core.utils import log
from specialk.models.mt_model import RNNModule, TransformerModule
from specialk.models.tokenizer import SentencePieceVocabulary
from specialk.models.transformer.torch.pytorch_transformer import (
    PyTorchTransformerModel,
    PyTorchTransformerModule,
)
from tests.tokenizer.test_tokenizer import VOCABULARY_SIZE

dirpath: Path = Path("tests/tokenizer/test_files")
tokenizer_path: Path = (
    PROJECT_DIR / "assets" / "tokenizer" / "sentencepiece" / "enfr.model"
)
dataset_path = PROJECT_DIR / "tests/test_files/datasets/en_fr.parquet"

BATCH_SIZE = 3
SEQUENCE_LENGTH = 100

torch.manual_seed(1337)


@pytest.fixture(scope="session", autouse=True)
def dataset() -> Dataset:
    return Dataset.from_pandas(pd.read_parquet(dataset_path))


@pytest.fixture(scope="session")
def spm_tokenizer() -> SentencePieceVocabulary:
    return SentencePieceVocabulary.from_file(tokenizer_path, max_length=SEQUENCE_LENGTH)


@pytest.fixture(scope="session")
def spm_dataloader(dataset: Dataset, spm_tokenizer: SentencePieceVocabulary):
    def tokenize(example):
        # perform tokenization at this stage.
        example[SOURCE] = spm_tokenizer.to_tensor(example[SOURCE])
        example[TARGET] = spm_tokenizer.to_tensor(example[TARGET])
        return example

    tokenized_dataset = dataset.with_format("torch").map(tokenize)
    dataloader = DataLoader(tokenized_dataset, batch_size=BATCH_SIZE, shuffle=True)
    return dataloader, spm_tokenizer


def test_rnn_inference(spm_dataloader):
    dataloader, tokenizer = spm_dataloader
    model = RNNModule(
        name="rnn_1",
        vocabulary_size=tokenizer.vocab_size,
        sequence_length=SEQUENCE_LENGTH,
    )
    model.model.PAD = tokenizer.PAD

    batch: dict = next(iter(dataloader))
    x: torch.Tensor = batch[SOURCE].squeeze(1)
    y: torch.Tensor = batch[TARGET].squeeze(1)

    m = model.model
    y_hat = m(x, y)

    # flatten tensors for loss function.
    y_hat: Float[Tensor, "batch_size*seq_length, vocab"] = y_hat.view(
        -1, y_hat.size(-1)
    )
    y: Int[Tensor, "batch_size*vocab"] = y.view(-1)

    log.info("Y Shapes", y_hat=y_hat.shape, y=y.shape)
    loss = F.cross_entropy(y_hat, y, ignore_index=model.model.PAD, reduction="sum")
    loss.backward()


def test_rnn_inference_mps(spm_dataloader):
    dataloader, tokenizer = spm_dataloader
    model = RNNModule(
        name="rnn_1",
        vocabulary_size=tokenizer.vocab_size,
        sequence_length=SEQUENCE_LENGTH,
    )
    device = "mps"
    model.model.to(device)
    model.model.PAD = tokenizer.PAD

    batch: dict = next(iter(dataloader))
    x: torch.Tensor = batch[SOURCE].squeeze(1).to(device)
    y: torch.Tensor = batch[TARGET].squeeze(1).to(device)

    m = model.model
    y_hat = m(x, y)

    # flatten tensors for loss function.
    y_hat: Float[Tensor, "batch_size*seq_length, vocab"] = y_hat.view(
        -1, y_hat.size(-1)
    )
    y: Int[Tensor, "batch_size*vocab"] = y.view(-1)

    log.info("Y Shapes", y_hat=y_hat.shape, y=y.shape)
    loss = F.cross_entropy(y_hat, y, ignore_index=model.model.PAD, reduction="sum")
    loss.backward()


def test_transformer_inference(spm_dataloader):
    dataloader, tokenizer = spm_dataloader
    model = TransformerModule(
        name="transformer_2",
        vocabulary_size=tokenizer.vocab_size,
        sequence_length=SEQUENCE_LENGTH,
        d_word_vec=16,
        d_model=16,
        d_inner=8,
        n_layers=2,
        n_head=2,
        d_k=3,
        d_v=3,
        dropout=0.1,
    )
    model.change_pos_enc_len(SEQUENCE_LENGTH)
    model.PAD = tokenizer.PAD
    model.eval()

    batch: dict = next(iter(dataloader))
    x: torch.Tensor = batch[SOURCE].squeeze(1)
    y: torch.Tensor = batch[TARGET].squeeze(1)

    len_x = torch.arange(SEQUENCE_LENGTH).repeat((BATCH_SIZE, 1)) + 1
    len_x = len_x.masked_fill((x == PAD), 0)
    len_y = torch.arange(SEQUENCE_LENGTH).repeat((BATCH_SIZE, 1)) + 1
    len_y = len_y.masked_fill((y == PAD), 0)

    m = model.model

    # forward pass
    enc_output, *_ = m.encoder(x, len_x)
    dec_output, *_ = m.decoder(y, len_y, x, enc_output)
    y_hat = m.generator(dec_output) * m.x_logit_scale

    y_hat: Float[Tensor, "batch_size*seq_length, vocab"] = y_hat.view(
        -1, y_hat.size(-1)
    )
    y: Int[Tensor, "batch_size*vocab"] = y.view(-1)
    _ = F.cross_entropy(y_hat, y, ignore_index=model.PAD, reduction="sum")


def test_pt_transformer_inference(spm_dataloader):
    dataloader, tokenizer = spm_dataloader
    model = PyTorchTransformerModule(
        name="transformer_1",
        vocabulary_size=tokenizer.vocab_size,
        sequence_length=SEQUENCE_LENGTH,
        dim_model=8,
        n_heads=2,
        num_encoder_layers=1,
        num_decoder_layers=1,
    )
    model.model: PyTorchTransformerModel
    model.PAD = 0
    model.eval()

    batch: dict = next(iter(dataloader))
    x: torch.Tensor = batch[SOURCE].squeeze(1)
    y: torch.Tensor = batch[TARGET].squeeze(1)
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
