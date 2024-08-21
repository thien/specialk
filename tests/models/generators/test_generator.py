from pathlib import Path

import pytest
import torch
from jaxtyping import Bool, Float
from torch import Tensor
from tqdm import tqdm

from specialk.core.constants import EOS, PAD, PROJECT_DIR, SEED, SOS
from specialk.core.utils import log
from specialk.models.generators.sampling import EncoderDecoderSampler
from specialk.models.transformer.torch.pytorch_transformer import (
    PyTorchTransformerModule,
)


@pytest.fixture(scope="session", autouse=True)
def module() -> PyTorchTransformerModule:
    checkpoint_path = next(
        Path(
            f"{PROJECT_DIR}/tb_logs/nmt_model_dummy/transformer_smol/version_0/checkpoints/"
        ).iterdir()
    )

    module = PyTorchTransformerModule.load_from_checkpoint(
        checkpoint_path=checkpoint_path
    )
    return module


@pytest.fixture(scope="session", autouse=True)
def sampler(module) -> EncoderDecoderSampler:
    sampler = EncoderDecoderSampler(
        module.model, module.tokenizer, module.decoder_tokenizer, module.device
    )
    return sampler


def test_generator(sampler):
    test_src_text = "Eine Gruppe von Männern lädt Baumwolle auf einen Lastwagen"
    generated_text = sampler.sample(test_src_text, top_p=0.1, seed=SEED)
    text = generated_text[0]
    # yes, the actual translation is "A group of men are loading cotton onto a truck"
    # but the model is only a dummy!
    assert text.lower().startswith("a group of men")


def test_generator_batch():
    test_src = [
        "Eine Gruppe von Männern lädt Baumwolle auf einen Lastwagen",
        "Ein Mann schläft in einem grünen Raum auf einem Sofa.",
    ]
    test_tgt = [
        "A group of men are loading cotton onto a truck",
        "A man sleeping in a green room on a couch.",
    ]
    # i haven't implemented this yet
    pass
