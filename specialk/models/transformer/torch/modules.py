from __future__ import annotations

from typing import Callable, List, Optional, Tuple, Union

import einops
import torch
import torch.nn.functional as F
from jaxtyping import Float, Int
from torch import Tensor, nn
from torch.nn.modules import Dropout, LayerNorm, Linear
from torch.nn.modules.transformer import (TransformerDecoder,
                                          TransformerDecoderLayer)


class KVCache(Tensor):
    """
    Key/Value Cache Tensor that we'll use to keep track of KV operations.
    of cached values so we reduce the amount of ops used in a forward pass.
    This is particularly useful in the decoder generation.

    To use:
        KVCacheTensor = Float[Tensor, "2 batch seq_len n_heads d_head"]

    There's 2 dimensions, representing Key and Value.
    """

    @classmethod
    def new_empty(
        cls, n_layers: int, n_heads: int, d_head: int, batch: int = 1
    ) -> KVCache:
        """Generate new KV cache tensor."""
        shape = (n_layers, 2, batch, 0, n_heads, d_head)
        return cls(*shape)

    @property
    def k(self) -> Tensor:
        """Return key tensor."""
        return self[:, 0]

    @property
    def v(self) -> Tensor:
        """Return value tensor."""
        return self[:, 1]

    @property
    def batch(self) -> int:
        """Return batch size."""
        return self.shape[2]

    @property
    def seq_len(self) -> int:
        """Return sequence length."""
        return self.shape[3]


class CachedAttention(nn.Module):
    """Self attention module."""

    IGNORE: Float[Tensor, ""]

    def __init__(self, n_heads: int, d_model: int, d_head: int, init_range: float):
        super().__init__()
        self.W_Q = nn.Parameter(torch.empty((n_heads, d_model, d_head)))
        self.W_K = nn.Parameter(torch.empty((n_heads, d_model, d_head)))
        self.W_V = nn.Parameter(torch.empty((n_heads, d_model, d_head)))
        self.W_O = nn.Parameter(torch.empty((n_heads, d_head, d_model)))
        self.b_Q = nn.Parameter(torch.zeros((n_heads, d_head)))
        self.b_K = nn.Parameter(torch.zeros((n_heads, d_head)))
        self.b_V = nn.Parameter(torch.zeros((n_heads, d_head)))
        self.b_O = nn.Parameter(torch.zeros((d_model)))
        nn.init.normal_(self.W_Q, std=init_range)
        nn.init.normal_(self.W_K, std=init_range)
        nn.init.normal_(self.W_V, std=init_range)
        nn.init.normal_(self.W_O, std=init_range)
        self.register_buffer("IGNORE", torch.tensor(-1e5, dtype=torch.float32))

    def forward(
        self,
        key: Float[Tensor, "batch posn d_model"],
        query: Float[Tensor, "batch posn d_model"],
        value: Float[Tensor, "batch posn d_model"],
        key_padding_mask: Optional[Tensor] = None,
        need_weights: bool = True,
        attn_mask: Optional[Tensor] = None,
        average_attn_weights: bool = True,
        is_causal: bool = False,
        kv_cache_entry: Optional[KVCache] = None,
        **kwargs,
    ) -> Tuple[Float[Tensor, "batch posn d_model"], Optional[KVCache]]:
        """
        Returns the result of applying attention layer to normlized_resid_pre, as well as
        the new cached key and value vectors (which we get from concatenating the old cached
        ones with the new key and value vectors).
        """

        q = (
            einops.einsum(
                key,
                self.W_Q,
                "batch posn d_model, n_heads d_model d_head -> batch posn n_heads d_head",
            )
            + self.b_Q
        )
        k = (
            einops.einsum(
                query,
                self.W_K,
                "batch posn d_model, n_heads d_model d_head -> batch posn n_heads d_head",
            )
            + self.b_K
        )
        v = (
            einops.einsum(
                value,
                self.W_V,
                "batch posn d_model, n_heads d_model d_head -> batch posn n_heads d_head",
            )
            + self.b_V
        )

        # If cache_entry is not None, this means we use the previous key and value vectors
        # Also we'll need to get a new cache entry which will be used later to construct a new cache
        if kv_cache_entry:
            k = torch.concat([kv_cache_entry[0], k], dim=1)
            v = torch.concat([kv_cache_entry[1], v], dim=1)
            new_kv_cache = torch.stack([k, v])
        else:
            new_kv_cache = None

        attn_scores = einops.einsum(
            q,
            k,
            "batch posn_Q n_heads d_head, batch posn_K n_heads d_head -> batch n_heads posn_Q posn_K",
        )
        attn_scores_masked = self.apply_causal_mask(attn_scores / self.cfg.d_head**0.5)
        attn_pattern = attn_scores_masked.softmax(-1)

        z = einops.einsum(
            v,
            attn_pattern,
            "batch posn_K n_heads d_head, batch n_heads posn_Q posn_K -> batch posn_Q n_heads d_head",
        )

        out = (
            einops.einsum(
                z,
                self.W_O,
                "batch posn_Q n_heads d_head, n_heads d_head d_model -> batch posn_Q d_model",
            )
            + self.b_O
        )

        return out, new_kv_cache

    def apply_causal_mask(
        self, attn_scores: Float[Tensor, "batch n_heads query_pos key_pos"]
    ) -> Float[Tensor, "batch n_heads query_pos key_pos"]:
        """
        Here, attn_scores have shape (batch, n_heads, query_pos, key_pos), where query_pos represents the
        new (non-cached) positions, and key_pos represent all the positions (cached and non-cached).

        So when we create our mask, the query indices and key indices will both go up to the same value
        (the full sequence length), but the query indices will start at >0.
        """
        new_seq_len, full_seq_len = attn_scores.shape[-2:]
        assert new_seq_len <= full_seq_len
        q_posn = einops.repeat(
            attn_scores.new_tensor(range(full_seq_len - new_seq_len, full_seq_len)),
            "q -> q k",
            k=full_seq_len,
        )
        k_posn = einops.repeat(
            attn_scores.new_tensor(range(full_seq_len)), "k -> q k", q=new_seq_len
        )
        mask = q_posn < k_posn
        attn_scores = attn_scores.masked_fill(mask, self.IGNORE)
        return attn_scores


class CachedTransformerDecoderLayer(TransformerDecoderLayer):
    def __init__(
        self,
        d_model: int,
        nhead: int,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        activation: Callable[[Tensor], Tensor] = F.relu,
        layer_norm_eps: float = 1e-5,
        batch_first: bool = False,
        norm_first: bool = False,
        bias: bool = True,
        dtype=None,
    ) -> None:
        self.self_attn = CachedAttention(
            d_head=d_model,
            nhead,
            dropout=dropout,
            batch_first=batch_first,
            bias=bias,
            dtype=dtype,
        )
        self.multihead_attn = CachedAttention(
            d_model,
            nhead,
            dropout=dropout,
            batch_first=batch_first,
            bias=bias,
            dtype=dtype,
        )
        # Implementation of Feedforward model
        self.linear1 = Linear(d_model, dim_feedforward, bias=bias, dtype=dtype)
        self.dropout = Dropout(dropout)
        self.linear2 = Linear(dim_feedforward, d_model, bias=bias, dtype=dtype)
        self.norm_first = norm_first
        self.norm1 = LayerNorm(d_model, eps=layer_norm_eps, bias=bias, dtype=dtype)
        self.norm2 = LayerNorm(d_model, eps=layer_norm_eps, bias=bias, dtype=dtype)
        self.norm3 = LayerNorm(d_model, eps=layer_norm_eps, bias=bias, dtype=dtype)
        self.dropout1 = Dropout(dropout)
        self.dropout2 = Dropout(dropout)
        self.dropout3 = Dropout(dropout)
        self.activation = activation

    def forward(
        self,
        tgt: Tensor,
        memory: Tensor,
        tgt_mask: Optional[Tensor] = None,
        memory_mask: Optional[Tensor] = None,
        tgt_key_padding_mask: Optional[Tensor] = None,
        memory_key_padding_mask: Optional[Tensor] = None,
        tgt_is_causal: bool = False,
        memory_is_causal: bool = False,
    ) -> Tensor:
        r"""Pass the inputs (and mask) through the decoder layer.

        Args:
            tgt: the sequence to the decoder layer (required).
            memory: the sequence from the last layer of the encoder (required).
            tgt_mask: the mask for the tgt sequence (optional).
            memory_mask: the mask for the memory sequence (optional).
            tgt_key_padding_mask: the mask for the tgt keys per batch (optional).
            memory_key_padding_mask: the mask for the memory keys per batch (optional).
            tgt_is_causal: If specified, applies a causal mask as ``tgt mask``.
                Default: ``False``.
                Warning:
                ``tgt_is_causal`` provides a hint that ``tgt_mask`` is
                the causal mask. Providing incorrect hints can result in
                incorrect execution, including forward and backward
                compatibility.
            memory_is_causal: If specified, applies a causal mask as
                ``memory mask``.
                Default: ``False``.
                Warning:
                ``memory_is_causal`` provides a hint that
                ``memory_mask`` is the causal mask. Providing incorrect
                hints can result in incorrect execution, including
                forward and backward compatibility.
        """
        # see Fig. 1 of https://arxiv.org/pdf/2002.04745v1.pdf

        x = tgt
        if self.norm_first:
            x = x + self._sa_block(
                self.norm1(x), tgt_mask, tgt_key_padding_mask, tgt_is_causal
            )
            x = x + self._mha_block(
                self.norm2(x),
                memory,
                memory_mask,
                memory_key_padding_mask,
                memory_is_causal,
            )
            x = x + self._ff_block(self.norm3(x))
        else:
            x = self.norm1(
                x + self._sa_block(x, tgt_mask, tgt_key_padding_mask, tgt_is_causal)
            )
            x = self.norm2(
                x
                + self._mha_block(
                    x, memory, memory_mask, memory_key_padding_mask, memory_is_causal
                )
            )
            x = self.norm3(x + self._ff_block(x))

        return x

    # self-attention block
    def _sa_block(
        self,
        x: Tensor,
        attn_mask: Optional[Tensor],
        key_padding_mask: Optional[Tensor],
        is_causal: bool = False,
    ) -> Tensor:
        x = self.self_attn(
            x,
            x,
            x,
            attn_mask=attn_mask,
            key_padding_mask=key_padding_mask,
            is_causal=is_causal,
            need_weights=False,
        )[0]
        return self.dropout1(x)

    # multihead attention block
    def _mha_block(
        self,
        x: Tensor,
        mem: Tensor,
        attn_mask: Optional[Tensor],
        key_padding_mask: Optional[Tensor],
        is_causal: bool = False,
    ) -> Tensor:
        x = self.multihead_attn(
            x,
            mem,
            mem,
            attn_mask=attn_mask,
            key_padding_mask=key_padding_mask,
            is_causal=is_causal,
            need_weights=False,
        )[0]
        return self.dropout2(x)

    # feed forward block
    def _ff_block(self, x: Tensor) -> Tensor:
        x = self.linear2(self.dropout(self.activation(self.linear1(x))))
        return self.dropout3(x)


class CachedTransformerDecoder(TransformerDecoder):
    def __init__(*args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(
        self,
        tgt: Tensor,
        memory: Tensor,
        tgt_mask: Optional[Tensor] = None,
        memory_mask: Optional[Tensor] = None,
        tgt_key_padding_mask: Optional[Tensor] = None,
        memory_key_padding_mask: Optional[Tensor] = None,
        tgt_is_causal: Optional[bool] = None,
        memory_is_causal: bool = False,
        kv_cache: Optional[KVCache] = None,
    ) -> Tuple[Tensor, Union[KVCache, None]]:
        r"""Pass the inputs (and mask) through the decoder layer in turn.

        Args:
            tgt: the sequence to the decoder (required).
            memory: the sequence from the last layer of the encoder (required).
            tgt_mask: the mask for the tgt sequence (optional).
            memory_mask: the mask for the memory sequence (optional).
            tgt_key_padding_mask: the mask for the tgt keys per batch (optional).
            memory_key_padding_mask: the mask for the memory keys per batch (optional).
            tgt_is_causal: If specified, applies a causal mask as ``tgt mask``.
                Default: ``None``; try to detect a causal mask.
                Warning:
                ``tgt_is_causal`` provides a hint that ``tgt_mask`` is
                the causal mask. Providing incorrect hints can result in
                incorrect execution, including forward and backward
                compatibility.
            memory_is_causal: If specified, applies a causal mask as
                ``memory mask``.
                Default: ``False``.
                Warning:
                ``memory_is_causal`` provides a hint that
                ``memory_mask`` is the causal mask. Providing incorrect
                hints can result in incorrect execution, including
                forward and backward compatibility.

        Shape:
            see the docs in :class:`~torch.nn.Transformer`.
        """
        output = tgt

        seq_len = _get_seq_len(tgt, self.layers[0].self_attn.batch_first)
        tgt_is_causal = _detect_is_causal_mask(tgt_mask, tgt_is_causal, seq_len)


        using_kv_cache = kv_cache is not None

        if using_kv_cache:
            kv_cache = [None for _ in range(self.cfg.n_layers)]

        new_kv_cache_entries: List[KVCache] = []
        for block, kv_cache_entry in zip(self.layers, kv_cache):
            output, kv_cache_entry = block(
                output,
                memory,
                tgt_mask=tgt_mask,
                memory_mask=memory_mask,
                tgt_key_padding_mask=tgt_key_padding_mask,
                memory_key_padding_mask=memory_key_padding_mask,
                tgt_is_causal=tgt_is_causal,
                memory_is_causal=memory_is_causal,
            )
            if using_kv_cache:
                new_kv_cache_entries.append(kv_cache_entry)

        if self.norm is not None:
            output = self.norm(output)


        if using_kv_cache:
            return output, KeyValueCache(torch.stack(new_kv_cache_entries))
        else:
            return output, None



class CachedTransformer(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        self.embed = Embed(cfg)
        self.pos_embed = PosEmbed(cfg)
        self.blocks = nn.ModuleList(
            [TransformerBlock(cfg) for _ in range(cfg.n_layers)]
        )
        self.ln_final = LayerNorm(cfg)
        self.unembed = Unembed(cfg)

    def forward(
        self,
        tokens: Int[Tensor, "batch seq_pos"],
        kv_cache: Optional[KVCache] = None,
    ) -> Tuple[Float[Tensor, "batch position d_vocab"], Union[KVCache, None]]:
        using_kv_cache = kv_cache is not None

        if using_kv_cache:
            # If using kv_cache, then we only need to pass forward the newest tokens
            # Remember to add positional offset!
            n_cached_tokens = kv_cache.seq_len
            tokens = tokens[:, n_cached_tokens:]
            residual = self.embed(tokens) + self.pos_embed(tokens, n_cached_tokens)
        else:
            # If not using cache, turn it into a list of None's (so we can iterate through it)
            kv_cache = [None for _ in range(self.cfg.n_layers)]
            residual = self.embed(tokens) + self.pos_embed(tokens)

        # Apply all layers, and create a (new) kv_cache from the key & value vectors
        new_kv_cache_entries: List[KVCache] = []
        for block, kv_cache_entry in zip(self.blocks, kv_cache):
            residual, kv_cache_entry = block(residual, kv_cache_entry)
            if using_kv_cache:
                new_kv_cache_entries.append(kv_cache_entry)

        logits = self.unembed(self.ln_final(residual))

        if using_kv_cache:
            return logits, KeyValueCache(torch.stack(new_kv_cache_entries))
        else:
            return logits, None
