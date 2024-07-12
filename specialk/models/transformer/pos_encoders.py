import torch.nn as nn
import torch
import numpy as np
from jaxtyping import Float


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
        self, x: Float[torch.Tensor, "batch seq_len embedding"]
    ) -> Float[torch.Tensor, "batch seq_len embedding"]:
        """Adds positional encoding to each item in the index.

        Returns:
            torch.Tensor: x with the positional encodings added to each value.
        """
        _, sequence_length, _ = x.shape
        print("pos enc", self.pos_enc.shape)
        print("pos enc min", self.pos_enc[:, :sequence_length, :].shape)
        return x + self.pos_enc[:, :sequence_length, :]
