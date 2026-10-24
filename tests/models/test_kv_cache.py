import pytest
import torch

from specialk.core.constants import PAD, SOS
from specialk.models.transformer.torch.pytorch_transformer import (
    PyTorchTransformerModel,
)

DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"


@pytest.mark.parametrize("norm_first", [False, True])
def test_decode_step_matches_decode(norm_first):
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
    model.eval()

    x = torch.randint(5, 50, (3, 7), device=DEVICE)
    x[0, 5:] = PAD  # padded source exercises memory_key_padding_mask
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
