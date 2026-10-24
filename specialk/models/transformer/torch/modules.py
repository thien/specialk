"""
Key/Value caching for incremental decoding with stock nn.TransformerDecoder.

Reuses the weights of nn.TransformerDecoderLayer (so existing checkpoints work),
but runs one new target token per step instead of re-running the whole prefix.

Cache tensors are batch-first so beams can index/repeat them like any other
per-item tensor:
    KVCache = Float[Tensor, "batch n_layers 2 n_heads seq d_head"]
where dim 2 holds (key, value).
"""

from typing import Optional, Tuple

import einops
import torch
import torch.nn.functional as F
from jaxtyping import Bool, Float
from torch import Tensor, nn

KVCache = Float[Tensor, "batch n_layers 2 n_heads seq d_head"]


def _project(
    mha: nn.MultiheadAttention, x: Float[Tensor, "batch seq d_model"], i: int
) -> Float[Tensor, "batch n_heads seq d_head"]:
    """Apply the i-th (0=Q, 1=K, 2=V) slice of mha's packed in-projection."""
    e = mha.embed_dim
    b = None if mha.in_proj_bias is None else mha.in_proj_bias[i * e : (i + 1) * e]
    out = F.linear(x, mha.in_proj_weight[i * e : (i + 1) * e], b)
    return einops.rearrange(out, "b s (h d) -> b h s d", h=mha.num_heads)


def _attend(
    mha: nn.MultiheadAttention,
    q: Float[Tensor, "batch n_heads q_pos d_head"],
    k: Float[Tensor, "batch n_heads k_pos d_head"],
    v: Float[Tensor, "batch n_heads k_pos d_head"],
    key_padding_mask: Optional[Bool[Tensor, "batch k_pos"]] = None,
) -> Float[Tensor, "batch q_pos d_model"]:
    # key_padding_mask is True on pads; sdpa's bool mask is True where we attend.
    mask = None if key_padding_mask is None else ~key_padding_mask[:, None, None, :]
    z = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
    return mha.out_proj(einops.rearrange(z, "b h s d -> b s (h d)"))


def cross_kv(
    decoder: nn.TransformerDecoder, memory: Float[Tensor, "batch src d_model"]
) -> KVCache:
    """Project encoder output into every layer's cross-attention K/V, once."""
    return torch.stack(
        [
            torch.stack(
                [_project(layer.multihead_attn, memory, i) for i in (1, 2)], dim=1
            )
            for layer in decoder.layers
        ],
        dim=1,
    )


def decoder_step(
    decoder: nn.TransformerDecoder,
    x: Float[Tensor, "batch 1 d_model"],
    memory_kv: KVCache,
    memory_key_padding_mask: Optional[Bool[Tensor, "batch src"]] = None,
    kv_cache: Optional[KVCache] = None,
) -> Tuple[Float[Tensor, "batch 1 d_model"], KVCache]:
    """Run one new (embedded) target position through the decoder.

    Mirrors nn.TransformerDecoderLayer.forward. No causal mask is needed since
    the single query is the newest position and may attend to every cached key.

    Returns the decoder output for the new position and the extended cache.
    """
    new_cache = []
    for n, layer in enumerate(decoder.layers):
        sa, ca = layer.self_attn, layer.multihead_attn
        assert sa.batch_first and sa._qkv_same_embed_dim and sa.bias_k is None

        def self_attn_block(h):
            k, v = _project(sa, h, 1), _project(sa, h, 2)
            if kv_cache is not None:
                k = torch.cat([kv_cache[:, n, 0], k], dim=-2)
                v = torch.cat([kv_cache[:, n, 1], v], dim=-2)
            new_cache.append(torch.stack([k, v], dim=1))
            return layer.dropout1(_attend(sa, _project(sa, h, 0), k, v))

        def cross_attn_block(h):
            q = _project(ca, h, 0)
            out = _attend(
                ca, q, memory_kv[:, n, 0], memory_kv[:, n, 1], memory_key_padding_mask
            )
            return layer.dropout2(out)

        if layer.norm_first:
            x = x + self_attn_block(layer.norm1(x))
            x = x + cross_attn_block(layer.norm2(x))
            x = x + layer._ff_block(layer.norm3(x))
        else:
            x = layer.norm1(x + self_attn_block(x))
            x = layer.norm2(x + cross_attn_block(x))
            x = layer.norm3(x + layer._ff_block(x))

    if decoder.norm is not None:
        x = decoder.norm(x)
    return x, torch.stack(new_cache, dim=1)
