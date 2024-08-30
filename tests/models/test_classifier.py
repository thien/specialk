from pathlib import Path

import pytest
import torch
from torch import nn
from torchmetrics.functional import accuracy

from specialk.core.utils import log
from specialk.models.classifier.models import CNNClassifier
from specialk.models.classifier.onmt.CNNModels import ConvNet
from specialk.models.tokenizer import WordVocabulary
from tests.models.fixtures import bpe_dataloader  # noqa: F401; noqa: F402
from tests.models.fixtures import bpe_tokenizer  # noqa: F401; noqa: F402
from tests.models.fixtures import dataset  # noqa: F401; noqa: F402
from tests.models.fixtures import word_dataloader  # noqa: F401; noqa: F402
from tests.models.fixtures import word_tokenizer  # noqa: F401; noqa: F402
from tests.tokenizer.test_tokenizer import VOCABULARY_SIZE

dirpath = "tests/tokenizer/test_files"
dev_path = (
    "/Users/t/Projects/datasets/political/political_data/democratic_only.dev.en.parquet"
)

BATCH_SIZE = 128
SEQUENCE_LENGTH = 100

torch.manual_seed(1337)


def test_model_inference(bpe_dataloader):  # noqa: F811
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

    # log.info("one hot",one_hot=one_hot_old[:, 0, :])

    # one_hot = torch.nn.functional.one_hot(x.T, num_classes=VOCABULARY_SIZE).float()

    # assert not torch.equal(one_hot, one_hot_old)

    # log.info("one_hots", old=one_hot_old, new=one_hot)

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


def test_load_model_from_checkpoint():
    path_classifier = Path(
        "/Users/t/Projects/specialk/assets/classifiers/legacy/cnn_classifier/"
    )
    for category in [
        "adversarial_political",
        "adversarial_publication",
        "naturalness_political",
    ]:
        path_checkpoint = path_classifier / category / f"{category}.ckpt"
        path_hyperparams = path_classifier / category / f"hyperparameters.yaml"
        path_tok = path_classifier / category / f"tokenizer"

        module = CNNClassifier.load_from_checkpoint(
            path_checkpoint, hparams_file=path_hyperparams
        )
        module.tokenizer = WordVocabulary.from_file(path_tok)

        text = ["Donald Trump!!!", "obama rules"] * 10
        batch_size = 3
        output = module.generate(text, batch_size)
        log.info("output", out=output, shape=output.shape)
        assert output.shape == torch.Size((len(text)))
        assert isinstance(output, torch.Tensor)
