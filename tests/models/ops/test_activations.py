import pytest
import torch

from specialk.models.utils.activations import SwiGLU


@pytest.mark.lightweight
def test_swiglu():
    # Example usage
    batch_size = 16
    input_dim = 21
    hidden_dim = 12
    output_dim = 5

    swiglu = SwiGLU(input_dim, hidden_dim, output_dim)
    input_tensor = torch.randn(batch_size, input_dim)  # Batch size of 32
    output = swiglu(input_tensor)
    assert output.shape == torch.Size((batch_size, output_dim))
