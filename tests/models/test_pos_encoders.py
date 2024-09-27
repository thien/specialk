import pytest
import torch

from specialk.models.transformer.pos_encoders import RotaryPositionalEncoder


@pytest.mark.lightweight
@torch.inference_mode()
def test_rope():
    # Parameters
    seq_len = 3
    dim = 4  # Small dimension for easy visualization

    # Create a simple increasing sequence
    x = torch.arange(1, seq_len * dim + 1, dtype=torch.float32).view((1, seq_len, dim))

    # Initialize the RotaryEmbedding
    rope = RotaryPositionalEncoder(dim_model=dim, max_seq_len=seq_len)

    # Apply rotary embedding
    y_hat = rope(x)

    assert y_hat.shape == x.shape

    actual = torch.Tensor(
        [
            [
                [1.0000, 2.0000, 3.0000, 4.0000],
                [-3.1888, 5.9197, 7.9895, 8.0596],
                [-13.7476, 9.7580, 3.6061, 12.1976],
            ]
        ]
    )
    assert torch.allclose(y_hat, actual, rtol=1e-4, atol=1e-4)


@pytest.mark.lightweight
@torch.inference_mode()
def test_rope_simple():
    # Reference test case from
    # https://colab.research.google.com/drive/11SKfzvMotuvvXNqY9qBpsD2RQX1PK7rP?usp=sharing#scrollTo=WyaC1bJYi5FH
    # Create a simple increasing sequence
    x = torch.tensor([[1, 2, 3, 4], [4, 5, 6, 7], [7, 8, 9, 10]])

    # Initialize the RotaryEmbedding
    rope = RotaryPositionalEncoder(dim_model=4, max_seq_len=3)

    # Apply rotary embedding
    y_hat = rope(x)

    assert y_hat.shape == x.shape

    actual = torch.tensor(
        [
            [1.0000, 2.0000, 3.0000, 4.0000],
            [-2.8876, 4.9298, 6.6077, 7.0496],
            [-11.0967, 7.7984, 2.6198, 10.1580],
        ]
    )
    assert torch.allclose(y_hat, actual, rtol=1e-4, atol=1e-4)
