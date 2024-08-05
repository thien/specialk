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

    def forward(
        self,
        query: Float[Tensor, "batch dim"],
        context: Float[Tensor, "batch seq_len dim"],
    ) -> tuple[Tensor, Tensor]:
        """
        Compute the global attention.

        Args:
            query: Tensor of shape (batch, dim)
            context: Tensor of shape (batch, seq_len, dim)

        Returns:
            tuple: (context_output, attention_weights)
                context_output: Tensor of shape (batch, dim)
                attention_weights: Tensor of shape (batch, seq_len)
        """
        target = self.linear_in(query).unsqueeze(2)  # shape: (batch, dim, 1)

        # Compute attention weights
        attn_weights: Float[Tensor, "batch seq_len"] = torch.bmm(
            context, target
        ).squeeze(2)

        # Apply mask if set
        if self.mask is not None:
            mask = self.mask
            if mask.shape[-1] > attn_weights.shape[-1]:
                if not self.warn_once:
                    self.warn_once = True
                    log.warn(
                        f"Attention mask {mask.shape} is greater than the "
                        f"attention matrix {attn_weights.shape}; truncating to fit. "
                        "This indicates a potential bug in your code."
                    )
                mask = mask[:, :, : attn_weights.shape[-1]]
            attn_weights.data.masked_fill_(mask.view(-1, mask.shape[-1]), -float("inf"))

        attn_weights = self.softmax(attn_weights)

        # Compute weighted context
        attn_applied = attn_weights.unsqueeze(1)  # shape: (batch, 1, seq_len)
        weighted_context = torch.bmm(attn_applied, context).squeeze(
            1
        )  # shape: (batch, dim)

        # Combine weighted context with input query
        combined_context = torch.cat((weighted_context, query), 1)
        context_output = self.tanh(self.linear_out(combined_context))

        return context_output, attn_weights
