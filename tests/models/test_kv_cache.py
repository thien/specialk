from types import SimpleNamespace

import pytest
import torch

from specialk.core.constants import EOS, PAD, SOS
from specialk.models.transformer.torch.pytorch_transformer import (
    PyTorchTransformerModel,
    TransformerBeamBatch,
    TransformerEncoderDecoderBeam,
)

DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"


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


def _src() -> torch.Tensor:
    x = torch.randint(5, 50, (3, 7), device=DEVICE)
    x[0, 5:] = PAD  # padded source exercises memory_key_padding_mask
    return x


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


@torch.inference_mode()
def test_beam_batch_cache_matches_uncached_beam():
    """Cached batch logits must survive the repeat/filter reshuffles of beams."""
    model, num_beams = _model(), 3
    memory, x_pad_mask = model.encode(_src())
    memory_kv = model.cross_kv(memory)
    batch = TransformerBeamBatch.create(
        [
            TransformerEncoderDecoderBeam(
                model=model,
                tokenizer=SimpleNamespace(EOS=EOS),
                logprob_sums=torch.zeros(num_beams, device=DEVICE),
                tokens=torch.full((num_beams, 1), SOS, device=DEVICE),
                memory=memory[i : i + 1].expand(num_beams, -1, -1),
                x_pad_mask=x_pad_mask[i : i + 1].expand(num_beams, -1),
                memory_kv=memory_kv[i : i + 1].expand(num_beams, *memory_kv.shape[1:]),
            )
            for i in range(memory.size(0))
        ]
    )
    for _ in range(4):
        log_probs = batch.get_logits()
        for beam, lp in zip(batch.beams, log_probs):
            # TransformerEncoderDecoderBeam.get_logits re-runs the full prefix.
            torch.testing.assert_close(lp, beam.get_logits(), atol=1e-4, rtol=1e-4)
        batch = TransformerBeamBatch.create(
            [b.generate(num_beams, log_logits=lp) for b, lp in zip(batch.beams, log_probs)]
        )
        batch, _ = batch.filter(num_beams)
