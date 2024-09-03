import torch
import torch.nn as nn
from torch import Tensor


class SwiGLU(nn.Module):
    """SwiGLU 'activation' function.

    It's more of a nn.layer, and not an activation function."""

    def __init__(
        self, in_features: int, hidden_features=None, out_features=None, bias=True
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.w1 = nn.Linear(in_features, hidden_features, bias=bias)
        self.w2 = nn.Linear(in_features, hidden_features, bias=bias)
        self.w3 = nn.Linear(hidden_features, out_features, bias=bias)

    def swish(self, x: Tensor) -> Tensor:
        """just do it"""
        return x * torch.sigmoid(x)

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass of SwiGLU"""
        x1, x2 = self.w1(x), self.w2(x)
        hidden = self.swish(x1) * x2
        return self.w3(hidden)
