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
from specialk.models.tokenizer import SentencePieceVocabulary, WordVocabulary
from specialk.models.transformer.torch.pytorch_transformer import (
    PyTorchTransformerModel,
    PyTorchTransformerModule,
)

dirpath: Path = Path("tests/tokenizer/test_files")
bilingual_tokenizer_path: Path = (
    PROJECT_DIR / "assets" / "tokenizer" / "sentencepiece" / "enfr_small.model"
)
monolingual_tokenizer_en: Path = (
    PROJECT_DIR / "assets" / "tokenizer" / "en_small_word_moses"
)

monolingual_tokenizer_de: Path = (
    PROJECT_DIR / "assets" / "tokenizer" / "de_small_word_moses"
)
dataset_path = PROJECT_DIR / "tests/test_files/datasets/en_fr.parquet"

BATCH_SIZE = 7
SEQUENCE_LENGTH = 50

torch.manual_seed(1337)


@pytest.fixture(scope="session", autouse=True)
def dataset() -> Dataset:
    return Dataset.from_pandas(pd.read_parquet(dataset_path))


@pytest.fixture(scope="session")
def spm_tokenizer() -> SentencePieceVocabulary:
    return SentencePieceVocabulary.from_file(
        bilingual_tokenizer_path, max_length=SEQUENCE_LENGTH
    )


@pytest.fixture(scope="session")
def en_word_tokenizer() -> WordVocabulary:
    return WordVocabulary.from_file(monolingual_tokenizer_en)


@pytest.fixture(scope="session")
def de_word_tokenizer() -> WordVocabulary:
    return WordVocabulary.from_file(monolingual_tokenizer_de)


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


@pytest.fixture(scope="session")
def word_dataloader_separate(
    dataset: Dataset,
    de_word_tokenizer: WordVocabulary,
    en_word_tokenizer: WordVocabulary,
):
    def tokenize(example):
        # perform tokenization at this stage.
        example[SOURCE] = de_word_tokenizer.to_tensor(example[SOURCE])
        example[TARGET] = en_word_tokenizer.to_tensor(example[TARGET])
        return example

    tokenized_dataset = dataset.with_format("torch").map(tokenize)
    dataloader = DataLoader(tokenized_dataset, batch_size=BATCH_SIZE, shuffle=True)
    return dataloader, de_word_tokenizer, en_word_tokenizer


def test_rnn_inference_separate_tokenizers(word_dataloader_separate):
    dataloader, src_tokenizer, tgt_tokenizer = word_dataloader_separate
    assert src_tokenizer.vocab_size != tgt_tokenizer.vocab_size
    model = RNNModule(
        name="rnn_1",
        vocabulary_size=src_tokenizer.vocab_size,
        decoder_vocabulary_size=tgt_tokenizer.vocab_size,
        sequence_length=SEQUENCE_LENGTH,
        tokenizer=src_tokenizer,
        decoder_tokenizer=tgt_tokenizer,
    )
    model.model.PAD = src_tokenizer.PAD

    batch: dict = next(iter(dataloader))
    x: torch.Tensor = batch[SOURCE].squeeze(1)
    y: torch.Tensor = batch[TARGET].squeeze(1)

    m = model.model
    y_hat = m(x, y)

    assert y_hat.shape[-1] == tgt_tokenizer.vocab_size

    # flatten tensors for loss function.
    y_hat: Float[Tensor, "batch_size*seq_length, vocab"] = y_hat.view(
        -1, y_hat.size(-1)
    )
    y: Int[Tensor, "batch_size*vocab"] = y.view(-1)

    log.info("Y Shapes", y_hat=y_hat.shape, y=y.shape)
    loss = F.cross_entropy(y_hat, y, ignore_index=model.model.PAD, reduction="sum")
    loss.backward()


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
    y_hat: Float[Tensor, "batch_size vocab"] = y_hat.view(-1, y_hat.size(-1))
    y: Int[Tensor, "batch_size"] = y.view(-1)

    assert y_hat.shape == torch.Size(
        [BATCH_SIZE * SEQUENCE_LENGTH, tokenizer.vocab_size]
    )
    assert y.shape == torch.Size([BATCH_SIZE * SEQUENCE_LENGTH])

    loss = F.cross_entropy(y_hat, y, ignore_index=model.model.PAD, reduction="sum")
    loss.backward()


def test_rnn_inference_brnn(spm_dataloader):
    dataloader, tokenizer = spm_dataloader

    model = RNNModule(
        name="rnn_1",
        vocabulary_size=tokenizer.vocab_size,
        brnn=True,
        sequence_length=SEQUENCE_LENGTH,
    )
    model.model.PAD = tokenizer.PAD

    batch: dict = next(iter(dataloader))
    x: torch.Tensor = batch[SOURCE].squeeze(1)
    y: torch.Tensor = batch[TARGET].squeeze(1)

    m = model.model
    y_hat = m(x, y)

    log.info("yhat", yhat=y_hat.shape)

    # flatten tensors for loss function.
    y_hat: Float[Tensor, "batch_size vocab"] = y_hat.view(-1, y_hat.size(-1))
    y: Int[Tensor, "batch_size"] = y.view(-1)

    assert y_hat.shape == torch.Size(
        [BATCH_SIZE * SEQUENCE_LENGTH, tokenizer.vocab_size]
    )
    assert y.shape == torch.Size([BATCH_SIZE * SEQUENCE_LENGTH])

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

    len_x = torch.arange(SEQUENCE_LENGTH).repeat((BATCH_SIZE, 1))
    len_x = len_x.masked_fill((x == PAD), 0)
    len_y = torch.arange(SEQUENCE_LENGTH).repeat((BATCH_SIZE, 1))
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


def test_transformer_inference_separate_tokenizers(word_dataloader_separate):
    dataloader, src_tokenizer, tgt_tokenizer = word_dataloader_separate
    assert src_tokenizer.vocab_size != tgt_tokenizer.vocab_size
    model = TransformerModule(
        name="transformer_2",
        vocabulary_size=src_tokenizer.vocab_size,
        decoder_vocabulary_size=tgt_tokenizer.vocab_size,
        tokenizer=src_tokenizer,
        decoder_tokenizer=tgt_tokenizer,
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
    model.PAD = src_tokenizer.PAD
    model.eval()

    batch: dict = next(iter(dataloader))
    x: torch.Tensor = batch[SOURCE].squeeze(1)
    y: torch.Tensor = batch[TARGET].squeeze(1)

    len_x = torch.arange(SEQUENCE_LENGTH).repeat((BATCH_SIZE, 1))
    len_x = len_x.masked_fill((x == PAD), 0)
    len_y = torch.arange(SEQUENCE_LENGTH).repeat((BATCH_SIZE, 1))
    len_y = len_y.masked_fill((y == PAD), 0)

    m = model.model

    # forward pass
    enc_output, *_ = m.encoder(x, len_x)
    dec_output, *_ = m.decoder(y, len_y, x, enc_output)
    y_hat = m.generator(dec_output) * m.x_logit_scale

    assert y_hat.shape[-1] == tgt_tokenizer.vocab_size


def test_pt_transformer_inference(spm_dataloader):
    dataloader, tokenizer = spm_dataloader
    module = PyTorchTransformerModule(
        name="transformer_1",
        vocabulary_size=tokenizer.vocab_size,
        tokenizer=tokenizer,
        sequence_length=SEQUENCE_LENGTH,
        dim_model=8,
        n_heads=2,
        num_encoder_layers=1,
        num_decoder_layers=1,
    )
    module.eval()
    model: PyTorchTransformerModel = module.model
    model.PAD = 0

    batch: dict = next(iter(dataloader))
    x: torch.Tensor = batch[SOURCE].squeeze(1)
    y: torch.Tensor = batch[TARGET].squeeze(1)

    # forward pass
    y_hat: Float[Tensor, "batch seq_length vocab"]
    y_hat = model.forward(x, y)

    # flatten tensors for loss function.
    y_hat: Float[Tensor, "batch_size*seq_length, vocab"] = y_hat.view(
        -1, y_hat.size(-1)
    )
    y: Int[Tensor, "batch_size*vocab"] = y.view(-1)
    _ = F.cross_entropy(y_hat, y, ignore_index=model.PAD, reduction="sum")
