import torch

from specialk.models.utils.activations import SwiGLU


def test_swiglu():
    # Example usage
    batch_size = 32
    input_dim = 128
    hidden_dim = 256
    output_dim = 64

    swiglu = SwiGLU(input_dim, hidden_dim, output_dim)
    input_tensor = torch.randn(batch_size, input_dim)  # Batch size of 32
    output = swiglu(input_tensor)
    assert output.shape == torch.Size((batch_size, output_dim))
