from typing import Optional

import einops
import numpy as np
import torch
import torch.nn as nn
from jaxtyping import Float
from torch import Tensor

from specialk.core.utils import deprecated


class PositionalEncoder(nn.Module):
    def __init__(self, dim_model: int, max_seq_length: int):
        """Implementation of the Positional Encoder used
        in "Attention is All You Need".

        Args:
            dim_model (int): Dimension of the positional encoding (same as emb.)
            max_seq_length (int): Maximum length to compute positional encodings
                against.
        """
        super(PositionalEncoder, self).__init__()
        pos_enc = torch.zeros((max_seq_length, dim_model))
        position = torch.arange(max_seq_length, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, dim_model, 2).float() * -(np.log(10000.0) / dim_model)
        )

        # sins in every even index, cos in every odd index.
        pos_enc[:, 0::2] = torch.sin(position * div_term)
        pos_enc[:, 1::2] = torch.cos(position * div_term)

        # we don't want to keep gradients (these aren't model params).
        self.register_buffer("pos_enc", pos_enc.unsqueeze(0))

    def forward(
        self, x: Float[Tensor, "batch seq_len embedding"]
    ) -> Float[Tensor, "batch seq_len embedding"]:
        """Adds positional encoding to each item in the index.

        Returns:
            Tensor: x with the positional encodings added to each value.
        """
        _, sequence_length, _ = x.shape
        return x + self.pos_enc[:, :sequence_length, :]


class RotaryPositionalEncoder(nn.Module):
    def __init__(self, dim_model: int, max_seq_len: int = 512, base: int = 10000):
        """Implementation of Rotary Positional Encodings (RoPE),
        used in RoFormer.

        From Eluther (https://blog.eleuther.ai/rotary-embeddings/):
        > Unlike standard positional embeddings, however, rotary embeddings must
          be applied at every layer. As large transformer models are typically
          dominated by matrix multiplies, we find that the overall overhead
          remains negligible. With fusion, we find that rotary embeddings
          impose a 1-3% overhead across a range of transformer sizes.

        Args:
            dim_model (int): Embedding dimension. This is usually set to the
                dim. of each head in the attention module (i.e. `embed_dim//n_heads`).
            max_seq_len (int): Maximum expected sequence length for the
                model, if exceeded the cached freqs will be recomputed.
            base (int): The base for the geometric progression used to compute
            the rotation angles.
        """
        super().__init__()
        self.base = base
        self.d = dim_model
        assert dim_model % 2 == 0, (
            "Dimension must be divisible by 2; "
            f"current dim_model={dim_model} does not."
        )
        self.max_seq_len = max_seq_len
        self._update_cos_sin_cache(max_seq_len)

    def _update_cos_sin_cache(self, seq_len: int):
        """Builds cache.

        This is regenerated if max_seq_len is greater than the current length.
        """
        if hasattr(self, "cos_cached"):
            if seq_len <= self.cos_cached.shape[0]:
                return

        # THETA = 10,000^(-2*i/d) or 1/10,000^(2i/d)
        theta = 1.0 / (self.base ** (torch.arange(0, self.d, 2) / self.d))

        seq_idx = torch.arange(seq_len).float()  # Position Index -> [0,1,2...seq-1]

        idx_theta = einops.einsum(seq_idx, theta, "n, d -> n d")
        # Cache the computation of cos and sin
        self.register_buffer("cos_cached", idx_theta.cos(), persistent=False)
        self.register_buffer("sin_cached", idx_theta.sin(), persistent=False)

    @deprecated
    @staticmethod
    def rotate(x: Tensor) -> Tensor:
        """
        Perform Rotation on complex plane.

        For instance, If you have an input tensor of shape [1, seq_len=5, d_model=4]:

        tensor([[[  1,   2,   3,   4],
                 [  5,   6,   7,   8],
                 [  9,  10,  11,  12],
                 [ 13,  14,  15,  16],
                 [ 17,  18,  19,  20]]])

        The rotation will look like this:

        tensor([[[ -2,  -4,   1,   3],
                 [ -6,  -8,   5,   7],
                 [-10, -12,   9,  11],
                 [-14, -16,  13,  15],
                 [-18, -20,  17,  19]]])

        This operation effectively creates a tensor that represents a
        90-degree rotation of the original tensor in a complex plane.
        """
        return torch.cat([-x[..., 1::2], x[..., ::2]], dim=-1)

    def forward(
        self, x: Float[torch.Tensor, "... s d"], seq_len: Optional[int] = None
    ) -> Float[torch.Tensor, "... s d"]:
        """
        Apply rotary embeddings to the input tensor.

        This method directly applies the rotation to the input tensor, operating on its
        last two dimensions (sequence length and embedding dimension).

        Args:
            x (Float[torch.Tensor, "... s d"]): Input tensor. The last two dimensions
                should be (seq_len, dim).

        Returns:
            Float[torch.Tensor, "... s d"]: Tensor with rotary embeddings applied.
        """
        if not seq_len:
            seq_len = x.shape[-2]

        if seq_len > self.max_seq_len:
            self._update_cos_sin_cache(seq_len)

        # Make sure we only use the cached values up to our sequence length
        cos = self.cos_cached[:seq_len, :]
        sin = self.sin_cached[:seq_len, :]

        # Reshape to match the input tensor's last two dimensions
        cos = cos.view((1,) * (x.ndim - 2) + cos.shape)
        sin = sin.view((1,) * (x.ndim - 2) + sin.shape)

        # Apply the rotation
        x1, x2 = x.chunk(2, dim=-1)

        # (a + bi) * (cos θ + i sin θ) = (a cos θ - b sin θ)
        #                             + i(b cos θ + a sin θ)
        rx1 = x1 * cos - x2 * sin
        rx2 = x2 * cos + x1 * sin

        return torch.cat([rx1, rx2], dim=-1)
