import time

import pytest
import torch

from specialk.core.constants import SOS
from specialk.models.transformer.torch.pytorch_transformer import (
    PyTorchTransformerModel,
)

DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"


def _sync():
    if DEVICE == "mps":
        torch.mps.synchronize()


@torch.inference_mode()
def _greedy(model, x, n_tokens, cached: bool) -> torch.Tensor:
    memory, x_pad_mask = model.encode(x)
    memory_kv, kv_cache = model.cross_kv(memory), None
    y = torch.full((x.size(0), n_tokens + 1), SOS, device=DEVICE)
    for i in range(1, n_tokens + 1):
        if cached:
            out, kv_cache = model.decode_step(
                y[:, i - 1 : i], memory_kv, x_pad_mask, kv_cache
            )
        else:
            out = model.decode(y[:, :i], memory, x_pad_mask=x_pad_mask)
        y[:, i] = model.generator(out[:, -1]).argmax(-1)
    return y


def _time(fn) -> float:
    fn()  # warmup
    _sync()
    start = time.perf_counter()
    fn()
    _sync()
    return time.perf_counter() - start


@pytest.mark.benchmark
@pytest.mark.parametrize("n_tokens", [25, 100, 200])
def test_kv_cache_speedup(n_tokens):
    """Run with: pytest -m benchmark -s tests/models/test_kv_cache_speed.py"""
    torch.manual_seed(0)
    # default (base-transformer) size, so timings resemble a real model.
    model = PyTorchTransformerModel(
        vocab_size=8000, max_seq_length=256, batch_first=True
    ).to(DEVICE).eval()
    x = torch.randint(5, 8000, (16, 40), device=DEVICE)

    uncached = _time(lambda: _greedy(model, x, n_tokens, cached=False))
    cached = _time(lambda: _greedy(model, x, n_tokens, cached=True))
    print(
        f"\n[{DEVICE}] {n_tokens} tokens: uncached {uncached:.3f}s, "
        f"cached {cached:.3f}s, speedup {uncached / cached:.2f}x"
    )

    assert torch.equal(
        _greedy(model, x, n_tokens, cached=False),
        _greedy(model, x, n_tokens, cached=True),
    )
    assert cached < uncached
