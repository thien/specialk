from typing import Optional, Tuple

import einops
import numpy as np
import torch
from jaxtyping import Float
from torch import Tensor, nn
from torch.nn import Parameter
from torch.nn.init import constant_, xavier_normal_, xavier_uniform_

from specialk.core.utils import log
from specialk.models.transformer.pos_encoders import RotaryPositionalEncoder


class Attention(nn.Module):
    """
    Vanilla Attention Implementation.

    Made to be interoperable as standard pytorch as reasonably possible.
    """

    IGNORE: Float[Tensor, ""]
    b_K: Optional[torch.Tensor]
    b_V: Optional[torch.Tensor]

    def __init__(
        self,
        d_head: int,
        num_heads: int,
        embed_dim: int,
        *,
        dropout=0.0,
        bias=False,
        add_bias_kv=True,
        add_zero_attn=False,
        kdim=None,
        vdim=None,
        batch_first=True,
    ):
        """
        MultiHead Attention implementation.

        This is mostly used for my sanity checking and understanding. There's also
        a lot less bloat in the code compared to PyTorch's nn.MultiHeadAttention,
        but there's no C level optimisation in this code.

        Args:
            d_head (int): Dimension of an individual attention head.
                Note that d_head * num_heads = embed_dim typically, (treat
                d_head is responsible for some set of channels.)
            embed_dim: Total dimension of the model.
            num_heads: Number of parallel attention heads. Note that
                ``embed_dim`` will be split across ``num_heads`` (i.e. each head
                will have dimension ``embed_dim // num_heads``).
            dropout: Dropout probability on ``attn_output_weights``.
                Default: ``0.0`` (no dropout).
            bias: If specified, adds bias to input / output projection layers.
                Default: ``True``.
            add_bias_kv: If specified, adds bias to the key and value sequences
                at dim=0. Default: ``False``.
            add_zero_attn: If specified, adds a new batch of zeros to the key
                and value sequences at dim=1.
                Default: ``False``.
            kdim: Total number of features for keys.
                Default: ``None`` (uses ``kdim=embed_dim``).
            vdim: Total number of features for values.
                Default: ``None`` (uses ``vdim=embed_dim``).
            batch_first: If ``True``, then the input and output tensors are provided
                as (batch, seq, feature). Default: ``False`` (seq, batch, feature).

        """
        super().__init__()

        if add_zero_attn or not batch_first:
            raise NotImplementedError

        if not (d_head * num_heads == embed_dim):
            log.warn(
                f"embed_dim={embed_dim} should be divisible by num_heads={d_head}."
            )

        self.kdim = embed_dim if kdim is None else kdim
        self.vdim = embed_dim if vdim is None else vdim
        self._qkv_same_embed_dim = self.kdim == embed_dim and self.vdim == embed_dim

        # In the reference implementation self.in_proj_weight is used (which contains
        # W_Q, W_K, W_V. This is technically more memory-efficient since it can end up
        # being one contiguous block of memory, but it's going to be chunked here.
        # So, this implementation is largely used for explanatory purposes.
        self.W_Q = Parameter(torch.empty((num_heads, embed_dim, d_head)))
        self.W_K = Parameter(torch.empty((num_heads, self.kdim, d_head)))
        self.W_V = Parameter(torch.empty((num_heads, self.vdim, d_head)))

        # self.W_O and self.b_O is functionally the same as self.out_proj
        self.W_O = Parameter(torch.empty((num_heads, d_head, embed_dim)))

        if bias:
            self.b_Q = Parameter(torch.zeros((num_heads, d_head)))

            if add_bias_kv:
                self.b_K = Parameter(torch.empty((num_heads, d_head)))
                self.b_V = Parameter(torch.empty((num_heads, d_head)))
            else:
                self.b_K = self.b_V = None

            # functionally equivalent as out_proj_bias.
            self.b_O = Parameter(torch.zeros((embed_dim)))
        else:
            self.b_Q = self.b_K = self.b_V = self.b_O = None

        self.register_buffer("IGNORE", torch.tensor(-1e5, dtype=torch.float32))
        self.dropout = torch.nn.Dropout(p=dropout) if dropout > 0.0 else None
        self._reset_parameters()

    def forward(
        self,
        x: Float[Tensor, "batch pos embed_dim"],
        is_causal=False,
        average_attn_weights=True,
    ) -> Tuple[
        Float[Tensor, "batch pos embed_dim"], Float[Tensor, "b num_heads q_pos k_pos"]
    ]:
        """
        Run Attention calculation.
        """
        d_head = self.W_Q.shape[-1]

        # get query, key, value vectors. Note that *seq_pos for q (q_pos)
        # and k (k_pos) are the same in self-attn*, but different in cross-attn.
        q: Float[Tensor, "b q_pos num_heads d_head"]
        k: Float[Tensor, "b k_pos num_heads d_head"]
        v: Float[Tensor, "b v_pos num_heads d_head"]

        q = einops.einsum(self.W_Q, x, "n_h m d_h, b p m -> b p n_h d_h")
        if self.b_Q is not None:
            q += self.b_Q
        k = einops.einsum(self.W_K, x, "n_h m d_h, b p m -> b p n_h d_h")
        if self.b_K is not None:
            k += self.b_K
        v = einops.einsum(self.W_V, x, "n_h m d_h, b p m -> b p n_h d_h")
        if self.b_V is not None:
            k += self.b_V

        # calculate attention scores; scale, mask, softmax.
        attn_scores: Float[Tensor, "b num_heads q_pos k_pos"]
        attn_scores = einops.einsum(q, k, "b p_q n h, b p_k n h -> b n p_q p_k")
        attn_scores /= np.sqrt(d_head)  # scale down.

        if is_causal:
            attn_scores = self.apply_causal_mask(attn_scores)

        attn_scores = attn_scores.softmax(dim=-1)

        if self.dropout is not None:
            attn_scores = self.dropout(attn_scores)

        # take weighted average of value vectors (weighted from attn_scores)
        z: Float[Tensor, "b q_pos num_heads d_head"]
        z = einops.einsum(v, attn_scores, "b p_k n h, b n p_q p_k -> b p_q n h")

        # return out. sum the num_heads.
        o: Float[Tensor, "b o_pos num_heads d_head"]  # n is summed in the process.
        o = einops.einsum(self.W_O, z, "n h m, b p_q n h -> b p_q m")

        if self.b_O is not None:
            o += self.b_O

        if average_attn_weights:
            # attention scores are averaged across the n_head dimension.
            attn_scores = attn_scores.mean(dim=1)

        return o, attn_scores

    def apply_causal_mask(
        self, attn_scores: Float[Tensor, "batch num_heads query_pos key_pos"]
    ) -> Float[Tensor, "batch num_heads query_pos key_pos"]:
        """
        Applies a causal mask to attention scores, and returns masked scores.
        It should look like this if dimensions of q, k == 4:

        tensor([[0., 1., 1., 1.],
                [0., 0., 1., 1.],
                [0., 0., 0., 1.],
                [0., 0., 0., 0.]])

        Offset it with the diagonal (you want [0,0]) to be False.
        """
        _, _, q_pos, k_pos = attn_scores.shape
        mask = torch.triu(
            torch.ones((q_pos, k_pos), device=attn_scores.device, dtype=torch.bool),
            diagonal=1,
        )
        return attn_scores.masked_fill(mask, self.IGNORE)

    def _reset_parameters(self):
        """Initialize weights with random values."""
        xavier_uniform_(self.W_Q)
        xavier_uniform_(self.W_K)
        xavier_uniform_(self.W_V)
        xavier_uniform_(self.W_O)
        if self.b_Q is not None:
            xavier_normal_(self.b_Q)
        if self.b_K is not None:
            xavier_normal_(self.b_K)
        if self.b_V is not None:
            xavier_normal_(self.b_V)
        if self.b_O is not None:
            constant_(self.b_O, 0.0)
