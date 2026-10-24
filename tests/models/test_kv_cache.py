import time
from contextlib import contextmanager
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

from specialk.core.constants import EOS, PAD, SEED, SOS
from specialk.models.transformer.torch.pytorch_transformer import (
    PyTorchTransformerModel,
    PyTorchTransformerModule,
)

DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
TOKENIZER = SimpleNamespace(EOS=EOS, PAD=PAD, SOS=SOS)
# pre-norm: a random post-norm decoder nearly ignores its target tokens, so
# cache bugs wouldn't change the sampled/beamed output.
SMALL = dict(dim_model=32, n_heads=4, dim_feedforward=64, norm_first=True)


def _model(norm_first=False) -> PyTorchTransformerModel:
    torch.manual_seed(0)
    model = PyTorchTransformerModel(
        vocab_size=50,
        max_seq_length=20,
        dim_model=32,
        n_heads=4,
        dim_feedforward=64,
        num_encoder_layers=2,
        num_decoder_layers=3,
        batch_first=True,
        norm_first=norm_first,
    ).to(DEVICE)
    return model.eval()


def _module(vocab_size=50, **kwargs) -> PyTorchTransformerModule:
    torch.manual_seed(0)
    module = PyTorchTransformerModule(
        vocabulary_size=vocab_size,
        sequence_length=256,
        tokenizer=TOKENIZER,
        **kwargs,
    )
    return module.to(DEVICE).eval()


def _src(batch=3, seq_len=7, vocab_size=50) -> torch.Tensor:
    x = torch.randint(5, vocab_size, (batch, seq_len), device=DEVICE)
    x[0, seq_len - 2 :] = PAD  # padded source exercises memory_key_padding_mask
    return x


@contextmanager
def uncached(model: PyTorchTransformerModel):
    """Make decode_step re-run the full prefix through decode().

    The "cache" becomes the token prefix, which is batch-first like real K/V,
    so generate() and beam search carry it through unchanged.
    """

    def decode_step(y, memory, x_pad_mask=None, kv_cache=None):
        tokens = y if kv_cache is None else torch.cat([kv_cache, y], dim=1)
        return model.decode(tokens, memory, x_pad_mask=x_pad_mask)[:, -1:], tokens

    with patch.object(model, "cross_kv", lambda memory: memory), patch.object(
        model, "decode_step", decode_step
    ):
        yield


def _generate(module, x, n_tokens):
    # seed once per run (not via seed=, which re-seeds every step and so draws
    # the same noise each step); both runs then see the same random stream.
    torch.manual_seed(SEED)
    return module.generate(x, max_len=n_tokens + 1, top_k=20)


def _beam_search(module, x, n_tokens):
    return module.beam_search(
        x, num_return_sequences=2, num_beams=4, max_new_tokens=n_tokens
    )


@torch.inference_mode()
def _rescore(model, x, tokens):
    """Length-normalised logprob of each returned beam, via the uncached decode()."""
    b, n, s = tokens.shape
    memory, x_pad_mask = model.encode(x)
    y = tokens.flatten(0, 1)
    out = model.decode(
        y[:, :-1],
        memory.repeat_interleave(n, 0),
        x_pad_mask=x_pad_mask.repeat_interleave(n, 0),
    )
    logp = model.generator(out).log_softmax(-1).gather(-1, y[:, 1:, None])[..., 0]
    is_eos = y == EOS
    length = torch.where(is_eos.any(-1), is_eos.int().argmax(-1) + 1, s)
    keep = torch.arange(1, s, device=y.device) < length[:, None]
    return (torch.where(keep, logp, 0).sum(-1) / length).view(b, n)


def _assert_same_beams(model, x, out, expected):
    """Hypotheses tied to ~1e-7 can resolve differently with or without the
    cache, so rather than comparing tokens, require the same scores and that
    each returned beam really earns its score under the uncached model."""
    (scores, tokens), (expected_scores, _) = out, expected
    torch.testing.assert_close(scores, expected_scores, atol=1e-4, rtol=1e-4)
    torch.testing.assert_close(_rescore(model, x, tokens), scores, atol=1e-4, rtol=1e-4)


@pytest.mark.parametrize("norm_first", [False, True])
def test_decode_step_matches_decode(norm_first):
    model = _model(norm_first)
    x = _src()
    y = torch.randint(5, 50, (3, 6), device=DEVICE)
    y[:, 0] = SOS

    with torch.inference_mode():
        memory, x_pad_mask = model.encode(x)
        memory_kv = model.cross_kv(memory)
        kv_cache = None
        for i in range(y.size(1)):
            out, kv_cache = model.decode_step(
                y[:, i : i + 1], memory_kv, x_pad_mask, kv_cache
            )
            expected = model.decode(y[:, : i + 1], memory, x_pad_mask=x_pad_mask)
            torch.testing.assert_close(out[:, 0], expected[:, -1], atol=1e-4, rtol=1e-4)
    assert kv_cache.shape == (3, 3, 2, 4, 6, 8)


def test_generate_matches_uncached():
    module, x = _module(**SMALL), _src()
    with uncached(module.model):
        expected_tokens = _generate(module, x, 15)
    torch.testing.assert_close(_generate(module, x, 15), expected_tokens)


def test_beam_search_matches_uncached():
    module, x = _module(**SMALL), _src()
    with uncached(module.model):
        expected = _beam_search(module, x, 10)
    _assert_same_beams(module.model, x, _beam_search(module, x, 10), expected)


def _timed(fn):
    fn()  # warmup
    if DEVICE == "mps":
        torch.mps.synchronize()
    start = time.perf_counter()
    out = fn()
    if DEVICE == "mps":
        torch.mps.synchronize()
    return time.perf_counter() - start, out


@pytest.mark.benchmark
@pytest.mark.parametrize("search", [_generate, _beam_search], ids=["sample", "beam"])
@pytest.mark.parametrize("n_tokens", [25, 100, 200])
def test_kv_cache_speedup(search, n_tokens):
    """Run with: pytest -m benchmark -s tests/models/test_kv_cache.py"""
    vocab_size = 8000
    # default (base-transformer) size, so timings resemble a real model.
    module = _module(vocab_size)
    x = _src(batch=16, seq_len=40, vocab_size=vocab_size)

    with uncached(module.model):
        t_uncached, expected = _timed(lambda: search(module, x, n_tokens))
    t_cached, out = _timed(lambda: search(module, x, n_tokens))
    print(
        f"\n[{DEVICE}] {search.__name__.strip('_')} {n_tokens} tokens: "
        f"uncached {t_uncached:.3f}s, cached {t_cached:.3f}s, "
        f"speedup {t_uncached / t_cached:.2f}x"
    )
    if search is _beam_search:
        _assert_same_beams(module.model, x, out, expected)
    else:
        torch.testing.assert_close(out, expected)


def test_beam_search_returns_distinct_beams():
    """All beams start as the same SOS-only sequence; expanding each of them
    used to fill the beam with copies of one hypothesis."""
    module, x = _module(**SMALL), _src()
    _, tokens = module.beam_search(
        x, num_return_sequences=3, num_beams=3, max_new_tokens=10
    )
    for item in tokens.tolist():
        assert len({tuple(seq) for seq in item}) == len(item)
