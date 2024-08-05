"""
Global attention takes a matrix and a query vector. It
then computes a parameterized convex combination of the matrix
based on the input query.
        H_1 H_2 H_3 ... H_n
          q   q   q       q
            |  |   |       |
              \ |   |      /
                      .....
                  \   |  /
                          a
Constructs a unit mapping.
    $$(H_1 + H_n, q) => (a)$$
    Where H is of `batch x n x dim` and q is of `batch x dim`.
    The full def is  $$\tanh(W_2 [(softmax((W_1 q + b_1) H) H), q] + b_2)$$.:
"""

import torch
import torch.nn as nn
from jaxtyping import Float
from torch import Tensor


class GlobalAttention(nn.Module):
    def __init__(self, dim: int):
        super(GlobalAttention, self).__init__()
        self.linear_in = nn.Linear(dim, dim, bias=False)
        self.softmax = nn.Softmax(dim=1)
        self.linear_out = nn.Linear(dim * 2, dim, bias=False)
        self.tanh = nn.Tanh()
        self.mask = None

    def applyMask(self, mask):
        self.mask = mask

    def forward(
        self,
        input: Float[Tensor, "batch dim"],
        context: Float[Tensor, "batch seq_len dim"],
    ):
        """
        input: batch x dim
        context: batch x sourceL x dim
        """
        targetT: Float[Tensor, "batch dim 1"] = self.linear_in(input).unsqueeze(
            2
        )  # batch x dim x 1
        # print("targetT.shape", targetT.shape)

        # Get attention
        attn: Float[Tensor, "batch seq_len"] = torch.bmm(context, targetT).squeeze(
            2
        )  # batch x sourceL
        # print("attn_shape", attn.shape)

        # apply mask
        if self.mask is not None:
            # print("mask.shape", self.mask.shape)
            attn.data.masked_fill_(
                self.mask.view(-1, self.mask.shape[-1]), -float("inf")
            )
            
        attn = self.softmax(attn)

        attn3 = attn.view(attn.size(0), 1, attn.size(1))  # batch x 1 x sourceL

        weightedContext = torch.bmm(attn3, context).squeeze(1)  # batch x dim
        contextCombined = torch.cat((weightedContext, input), 1)

        contextOutput = self.tanh(self.linear_out(contextCombined))

        return contextOutput, attn
