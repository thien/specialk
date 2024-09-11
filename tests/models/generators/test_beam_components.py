from dataclasses import dataclass
from typing import Optional

import einops
import pytest
import torch
from jaxtyping import Int

from specialk.core.utils import log

# Import the class to be tested
from specialk.models.transformer.torch.pytorch_transformer import (
    PyTorchTransformerModule,
    TransformerEncoderDecoderBeam,
)

from specialk.core import constants

torch.manual_seed(constants.SEED)


# Mock classes and functions to simulate the behavior of the actual classes
@dataclass
class MockModel:
    def encode(self, input_tokens):
        batch_size, seq_len = input_tokens.shape
        return torch.randn(batch_size, seq_len, 512), torch.randint(
            0, 2, (batch_size, seq_len)
        ).float()

    def decode(self, tokens, tgt_mask, memory, x_pad_mask):
        return torch.randn(tokens.shape[0], tokens.shape[1], 512)

    def generator(self, input_tensor):
        return torch.randn(input_tensor.shape[0], 1000)  # Assuming vocab size of 1000


@dataclass
class MockTokenizer:
    EOS: int = constants.EOS
    SOS: int = constants.SOS
    PAD: int = constants.PAD


@pytest.fixture
def mock_pytorch_transformer_module():
    class MockPyTorchTransformerModule(PyTorchTransformerModule):
        def __init__(self):
            self.model = MockModel()
            self.tokenizer = MockTokenizer()
            self.decoder_tokenizer = MockTokenizer()

    return MockPyTorchTransformerModule()


@pytest.fixture
def mock_beam():
    model = MockModel()
    tokenizer = MockTokenizer()
    batch_size = 10
    beam_size = 3
    seq_len = 10
    enc_seq = 8
    hidden_size = 512
    vocab_size = 1000

    return TransformerEncoderDecoderBeam(
        model=model,
        tokenizer=tokenizer,
        logprob_sums=torch.randn(batch_size * beam_size),
        tokens=torch.randint(0, vocab_size, (batch_size * beam_size, seq_len)),
        memory=torch.randn(batch_size * beam_size, enc_seq, hidden_size),
        x_pad_mask=torch.randint(0, 2, (batch_size * beam_size, enc_seq)).float(),
    )


def test_filter(mock_beam):
    num_beams = 2
    batch_size = mock_beam.logprob_sums.shape[0] // mock_beam.num_beams
    continuing, terminated = mock_beam.filter(num_beams)

    assert isinstance(continuing, TransformerEncoderDecoderBeam)
    assert isinstance(terminated, TransformerEncoderDecoderBeam)
    assert (
        continuing.logprob_sums.shape[0] + terminated.logprob_sums.shape[0]
        == num_beams * batch_size
    )


@pytest.fixture
def mock_beam_semi_completed():
    model = MockModel()
    tokenizer = MockTokenizer()
    batch_size = 1
    beam_size = 5
    seq_len = 10
    enc_seq = 8
    hidden_size = 512
    vocab_size = 1000

    # Create a tensor where some sequences contain the EOS token
    tokens: Int[torch.Tensor, "batch*beam seq"]
    tokens = torch.randint(
        low=10, high=vocab_size, size=(batch_size * beam_size, seq_len)
    )

    # Set EOS token for some sequences
    tokens[0, -1] = tokenizer.EOS  # First sequence in first batch
    tokens[2, -2] = tokenizer.EOS  # Last sequence in first batch
    tokens[4, -3] = tokenizer.EOS  # Middle sequence in second batch

    return TransformerEncoderDecoderBeam(
        model=model,
        tokenizer=tokenizer,
        logprob_sums=torch.randn(batch_size * beam_size),
        tokens=tokens,
        memory=torch.randn(batch_size * beam_size, enc_seq, hidden_size),
        x_pad_mask=torch.randint(0, 2, (batch_size * beam_size, enc_seq)).float(),
    )


def test_filter_with_terminated_sequences(mock_beam_semi_completed):
    beam = mock_beam_semi_completed
    # Modify the tokens to include some terminated sequences
    batch_size = beam.logprob_sums.shape[0] // beam.num_beams

    continuing, terminated = beam.filter()

    assert isinstance(continuing, TransformerEncoderDecoderBeam)
    assert isinstance(terminated, TransformerEncoderDecoderBeam)

    # Check that we have both continuing and terminated beams
    assert continuing.logprob_sums.shape[0] > 0
    assert terminated.logprob_sums.shape[0] > 0

    log.info(
        "n_beams",
        continuing=continuing.logprob_sums.shape[0],
        terminated=terminated.logprob_sums.shape[0],
    )

    # The total number of beams should be preserved
    assert (
        continuing.logprob_sums.shape[0] + terminated.logprob_sums.shape[0]
        == beam.num_beams
    )

    # Check that all sequences in the terminated beam actually contain the EOS token
    assert (terminated.tokens == beam.tokenizer.EOS).any(dim=1).all()

    # Check that none of the continuing sequences contain the EOS token
    assert not (continuing.tokens == beam.tokenizer.EOS).any()

    # Verify that we have the correct number of continuing beams per batch
    assert continuing.logprob_sums.shape[0] != batch_size * beam.num_beams


def test_new_beams(mock_beam):
    new_beam = mock_beam.new(
        mock_beam.logprob_sums, mock_beam.tokens, mock_beam.memory, mock_beam.x_pad_mask
    )
    assert isinstance(new_beam, TransformerEncoderDecoderBeam)
    assert torch.equal(new_beam.logprob_sums, mock_beam.logprob_sums)
    assert torch.equal(new_beam.tokens, mock_beam.tokens)
    assert torch.equal(new_beam.memory, mock_beam.memory)
    assert torch.equal(new_beam.x_pad_mask, mock_beam.x_pad_mask)


def test_get_logits(mock_beam):
    logits = mock_beam.get_logits()
    assert logits.shape == (
        mock_beam.tokens.shape[0],
        1000,
    )  # Assuming vocab size of 1000


def test_expand_encoder_outputs(mock_beam):
    tokens_per_beam = 2
    expanded = mock_beam.expand_encoder_outputs(tokens_per_beam)
    assert expanded.shape == (
        mock_beam.memory.shape[0] * tokens_per_beam,
        mock_beam.memory.shape[1],
        mock_beam.memory.shape[2],
    )


def test_expand_x_pad_masks(mock_beam):
    tokens_per_beam = 2
    expanded = mock_beam.expand_x_pad_masks(tokens_per_beam)
    assert expanded.shape == (
        mock_beam.x_pad_mask.shape[0] * tokens_per_beam,
        mock_beam.x_pad_mask.shape[1],
    )


def test_calculate_new_logprob_sums(mock_beam):
    tokens_per_beam = 2
    topk_log_probs = torch.randn(mock_beam.logprob_sums.shape[0], tokens_per_beam)
    new_logprob_sums = mock_beam._calculate_new_logprob_sums(
        topk_log_probs, tokens_per_beam
    )
    assert new_logprob_sums.shape == (
        mock_beam.logprob_sums.shape[0] * tokens_per_beam,
    )


def test_generate_new_tokens(mock_beam):
    tokens_per_beam = 2
    topk_token_idx = torch.randint(
        0, 1000, (mock_beam.tokens.shape[0], tokens_per_beam)
    )
    new_tokens = mock_beam._generate_new_tokens(topk_token_idx, tokens_per_beam)
    assert new_tokens.shape == (
        mock_beam.tokens.shape[0] * tokens_per_beam,
        mock_beam.tokens.shape[1] + 1,
    )


def test_getitem(mock_beam):
    idx = 1
    sub_beam = mock_beam[idx]
    assert isinstance(sub_beam, TransformerEncoderDecoderBeam)
    assert torch.equal(sub_beam.logprob_sums, mock_beam.logprob_sums[idx])
    assert torch.equal(sub_beam.tokens, mock_beam.tokens[idx])
    assert torch.equal(sub_beam.memory, mock_beam.memory[idx])
    assert torch.equal(sub_beam.x_pad_mask, mock_beam.x_pad_mask[idx])


def test_generate(mock_beam):
    tokens_per_beam = 2
    new_beam = mock_beam.generate(tokens_per_beam)
    assert isinstance(new_beam, TransformerEncoderDecoderBeam)
    assert (
        new_beam.logprob_sums.shape[0]
        == mock_beam.logprob_sums.shape[0] * tokens_per_beam
    )
    assert new_beam.tokens.shape[0] == mock_beam.tokens.shape[0] * tokens_per_beam
    assert new_beam.tokens.shape[1] == mock_beam.tokens.shape[1] + 1


# New test for beam_search method in PyTorchTransformerModule
def test_beam_search(mock_pytorch_transformer_module):
    # Set up input parameters
    batch_size = 2
    seq_len = 10
    input_tokens = torch.randint(0, 1000, (batch_size, seq_len))
    num_return_sequences = 2
    num_beams = 3
    max_new_tokens = 5
    length_penalty = 1.0
    no_repeat_ngram_size = None
    verbose = False

    # Perform beam search
    logprobs, logits = mock_pytorch_transformer_module.beam_search(
        input_tokens,
        num_return_sequences,
        num_beams,
        max_new_tokens,
        length_penalty,
        no_repeat_ngram_size,
        verbose,
    )

    assert logprobs.shape == torch.Size((batch_size, num_return_sequences))
    # it's +1 because of the BOS token.
    assert logits.shape == torch.Size(
        (batch_size, num_return_sequences, max_new_tokens + 1)
    )
