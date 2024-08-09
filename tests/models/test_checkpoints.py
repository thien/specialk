from __future__ import division

from pathlib import Path
from typing import Optional, Tuple, Union

import lightning.pytorch as pl
import pandas as pd
import torch

from specialk.core.constants import SOURCE, TARGET
from specialk.core.utils import check_torch_device, log
from specialk.models.mt_model import RNNModule
from specialk.models.transformer.torch.pytorch_transformer import (
    PyTorchTransformerModule as TransformerModule,
)
from tests.tokenizer.test_tokenizer import VOCABULARY_SIZE

DEVICE: str = check_torch_device()
SEQUENCE_LENGTH = 50
BATCH_SIZE = 5

from tests.models.fixtures import bpe_dataloader  # noqa: F401; noqa: F402
from tests.models.fixtures import bpe_tokenizer  # noqa: F401; noqa: F402
from tests.models.fixtures import dataset  # noqa: F401; noqa: F402
from tests.models.fixtures import word_dataloader  # noqa: F401; noqa: F402
from tests.models.fixtures import word_tokenizer  # noqa: F401; noqa: F402


def test_save_load_checkpoint(bpe_dataloader):
    model = RNNModule(
        name="rnn_1",
        vocabulary_size=VOCABULARY_SIZE,
        sequence_length=SEQUENCE_LENGTH,
    )

    batch: dict = next(iter(bpe_dataloader))
    x: torch.Tensor = batch[SOURCE].squeeze(1)
    y: torch.Tensor = batch[TARGET].squeeze(1)

    m = model.model
    y_hat = m(x, y)

    trainer = Trainer()
    trainer.fit(model)

    # generate temporary path to save checkpoint and load from.
    trainer.save_checkpoint(TEMP_PATH)

    model2 = RNNModule.load_from_checkpoint(TEMP_PATH)

    y_hat_2 = model2(x, y)

    assert y_hat == y_hat_2
