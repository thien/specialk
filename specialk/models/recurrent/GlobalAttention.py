"""
Global attention module that computes a parameterized convex combination of a matrix
based on an input query vector.

The attention mechanism can be visualized as:

    H_1 H_2 H_3 ... H_n
     q   q   q       q
       |  |   |       |
         \ |   |      /
                 .....
             \   |  /
                     a

It constructs a unit mapping:
    $$(H_1 + H_n, q) => (a)$$
Where H is of shape `batch x seq_len x dim` and q is of shape `batch x dim`.

The full definition is:
    $$\tanh(W_2 [(softmax((W_1 q + b_1) H) H), q] + b_2)$$
"""

import torch
import torch.nn as nn
from einops import einsum
from jaxtyping import Float
from torch import Tensor

from specialk.core.utils import log


class GlobalAttention(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.linear_in = nn.Linear(dim, dim, bias=False)
        self.softmax = nn.Softmax(dim=1)
        self.linear_out = nn.Linear(dim * 2, dim, bias=False)
        self.tanh = nn.Tanh()
        self.mask: Tensor | None = None
        self.warn_once = False
        self.debug_flag = True

    def forward(
        self,
        query: Float[Tensor, "batch dim"],
        context: Float[Tensor, "seq_len batch dim"],
    ) -> tuple[Tensor, Tensor]:
        """
        Compute the global attention.

        Args:
            query: Tensor of shape (batch, dim)
            context: Tensor of shape (seq_len, batch, dim)

        Returns:
            tuple: (context_output, attention_weights)
                context_output: Tensor of shape (batch, dim)
                attention_weights: Tensor of shape (batch, seq_len)
        """
        attn_scores = einsum(
            context,
            self.linear_in(query),
            "seq_len batch dim, batch dim -> batch seq_len",
        )

        # Apply mask if set
        if self.mask is not None:
            mask = self.mask
            if mask.shape[-1] > attn_scores.shape[-1]:
                if not self.warn_once:
                    self.warn_once = True
                    log.warn(
                        f"Attention mask {mask.shape} is greater than the "
                        f"attention matrix {attn_scores.shape}; truncating to fit. "
                        "This indicates a potential bug in your code."
                    )
                mask = mask[:, :, : attn_scores.shape[-1]]
            attn_scores = attn_scores.masked_fill_(
                mask.view(-1, mask.shape[-1]), -float("inf")
            )

        attn_scores = self.softmax(attn_scores)
        weighted_context = einsum(
            attn_scores,
            context,
            "batch seq_len, seq_len batch dim -> batch dim",
        )
        # Combine weighted context with input query
        combined_context = torch.cat((weighted_context, query), 1)
        context_output = self.tanh(self.linear_out(combined_context))

        return context_output, attn_scores
