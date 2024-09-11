from pathlib import Path

import pytest
from jaxtyping import Int
from torch import Tensor
import torch
from specialk.core.constants import EOS, PAD, PROJECT_DIR, SEED, SOS
from specialk.core.utils import log

# from specialk.models.generators.sampling import EncoderDecoderSampler
from specialk.models.transformer.torch.pytorch_transformer import (
    PyTorchTransformerModule,
)
torch.manual_seed(SEED)

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


def test_generator_model(module):
    test_src_text = "Eine Gruppe von Männern lädt Baumwolle auf einen Lastwagen"
    tokens = module.tokenizer.to_tensor(test_src_text).to("mps")
    log.info("Tokens:", tokens=tokens, shape=tokens.shape)
    y_tokens = module.generate(tokens, top_p=0.2, seed=SEED)
    text = module.decoder_tokenizer.detokenize(y_tokens, specials=False)
    assert text[0].lower().startswith("a group of men")


def test_generator_beam(module):
    # "A group of men load cotton onto a truck"
    test_src_text = "Eine Gruppe von Männern lädt Baumwolle auf einen Lastwagen"
    tokens: Int[Tensor, "batch seq_len"]
    tokens = module.tokenizer.to_tensor(test_src_text).to("mps")
    log.info("Tokens:", tokens=tokens, shape=tokens.shape)
    y_hat = module.beam_search(
        tokens,
        num_return_sequences=1,
        num_beams=10,
        max_new_tokens=50,
        no_repeat_ngram_size=3,
        verbose=True,
    )
    log.info("y_hat", y_hat=y_hat)
    _, output_tokens = y_hat
    text = module.decoder_tokenizer.detokenize(output_tokens[:, 0, :], specials=False)
    log.info("output", y_hat=text)
    assert text[0].lower().startswith("a group of men")


def test_generator_beam_batch(module):
    test_src_text = [
        "Eine Gruppe von Männern lädt Baumwolle auf einen Lastwagen",
        "Ein Mann schläft in einem grünen Raum auf einem Sofa.",
    ]
    test_tgt_text = [
        "A group of men",
        "A man is",
    ]
    tokens: Int[Tensor, "batch seq_len"]
    tokens = module.tokenizer.to_tensor(test_src_text).to("mps")
    log.info("Tokens:", tokens=tokens, shape=tokens.shape)

    n_return_seq = 2
    y_hat = module.beam_search(
        tokens,
        num_return_sequences=n_return_seq,
        num_beams=5,
        max_new_tokens=30,
        no_repeat_ngram_size=5,
        verbose=True,
    )
    y_hat_scores, y_hat_tensors = y_hat

    assert y_hat_scores.shape[0] == y_hat_tensors.shape[0]
    assert y_hat_scores.shape[0] == len(test_src_text)
    assert y_hat_scores.shape[1] == y_hat_tensors.shape[1]
    assert y_hat_scores.shape[1] == n_return_seq

    log.info("y_hat", y_hat=y_hat)
    text = module.decoder_tokenizer.detokenize(y_hat_tensors[:, 0, :], specials=False)

    log.info("output", y_hat=text)
    for pred, actual in zip(text, test_tgt_text):
        assert pred.lower().startswith(actual.lower())


# @pytest.fixture(scope="session", autouse=True)
# def sampler(module) -> LanguageModelSampler:
#     sampler = LanguageModelSampler(
#         module.model, module.tokenizer, module.decoder_tokenizer, module.device
#     )
#     return sampler

# def test_generator(sampler):
#     test_src_text = "Eine Gruppe von Männern lädt Baumwolle auf einen Lastwagen"
#     generated_text = sampler.sample(test_src_text, top_p=0.1, seed=SEED)
#     text = generated_text[0]
#     # yes, the actual translation is "A group of men are loading cotton onto a truck"
#     # but the model is only a dummy!
#     assert text.lower().startswith("a group of men")
#
#
# def test_generator_batch(sampler):
#     test_src = [
#         "Eine Gruppe von Männern lädt Baumwolle auf einen Lastwagen",
#         "Ein Mann schläft in einem grünen Raum auf einem Sofa.",
#     ]
#     test_tgt = [
#         "A group of men",
#         "A man is sleeping",
#     ]
#     generated_text = sampler.sample(test_src, top_p=0.1, seed=SEED)
#     for expected, actual in zip(generated_text, test_tgt):
#         assert expected.lower().startswith(actual.lower())
