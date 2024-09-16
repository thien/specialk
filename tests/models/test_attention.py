import numpy as np
import torch
import torch.nn as nn
from jaxtyping import Float
from torch import Tensor

from specialk.core import constants
from specialk.core.utils import log
from specialk.models.transformer.torch.attention import Attention

torch.manual_seed(constants.SEED)
np.random.seed(constants.SEED)


@torch.inference_mode()
def test_reference_attention_n_heads_1():
    embed_dim = 8
    num_heads = 1
    seq_len = 3
    head_dim = embed_dim // num_heads

    # These are the default values, but w
    # kdim = embed_dim
    # vdim = embed_dim

    mha = nn.MultiheadAttention(
        embed_dim=embed_dim,
        num_heads=num_heads,
        dropout=0,
        bias=False,
        add_bias_kv=False,
        add_zero_attn=False,
    )  # kdim=kdim, vdim=vdim

    x = torch.rand(seq_len, embed_dim)

    # Self-attention
    # These are the reference tensors I want to obtain

    attn_output, attn_output_weights = mha(x, x, x)

    assert mha.in_proj_weight.shape == torch.Size([embed_dim * 3, embed_dim])
    assert mha.out_proj.weight.shape == torch.Size([embed_dim, embed_dim])

    wq, wk, wv = torch.split(
        mha.in_proj_weight, [embed_dim, embed_dim, embed_dim], dim=0
    )

    q = torch.matmul(x, wq.T)
    k = torch.matmul(x, wk.T)
    v = torch.matmul(x, wv.T)

    assert k.shape == torch.Size([seq_len, embed_dim])
    # x*W_q.T -> q, x*W_k.T -> k, x*W_v.T -> v
    # q @ k^T -> softmax -> attn_output_weights
    # (attn_output_weights * v)*out_proj.T -> attn_output

    scale = 1.0 / head_dim**0.5
    s = torch.softmax(q @ k.T * scale, dim=-1)

    torch.allclose(attn_output_weights, s)

    y = attn_output_weights @ v
    y = y.reshape(seq_len * 1, embed_dim)
    y = y @ mha.out_proj.weight.T
    y = y.reshape(seq_len, 1, embed_dim)
    y = y.squeeze(1)

    torch.allclose(attn_output, y)


@torch.inference_mode()
def test_reference_attention_n_heads_4_and_batched():
    embed_dim = 20
    num_heads = 4
    seq_len = 3
    bs = 2
    head_dim = embed_dim // num_heads

    # These are the default values, but w
    # kdim = embed_dim
    # vdim = embed_dim

    mha = nn.MultiheadAttention(
        embed_dim=embed_dim,
        num_heads=num_heads,
        dropout=0,
        bias=False,
        add_bias_kv=False,
        add_zero_attn=False,
    )  # kdim=kdim, vdim=vdim

    assert mha.in_proj_weight.shape == torch.Size([embed_dim * 3, embed_dim])
    assert mha.out_proj.weight.shape == torch.Size([embed_dim, embed_dim])

    x = torch.rand(seq_len, bs, embed_dim)

    attn_output, attn_output_weights = mha(x, x, x)

    assert attn_output.shape == torch.Size([seq_len, bs, embed_dim])
    assert attn_output_weights.shape == torch.Size([bs, seq_len, seq_len])

    wq, wk, wv = torch.split(
        mha.in_proj_weight, [embed_dim, embed_dim, embed_dim], dim=0
    )

    q = torch.matmul(x, wq.T)
    k = torch.matmul(x, wk.T)
    v = torch.matmul(x, wv.T)

    assert k.shape == torch.Size([seq_len, bs, embed_dim])

    # It is not necessary to reshape the input, this is already
    # how matmul works for higher dimensions
    torch.allclose(
        x @ wq.T,
        (x.reshape(seq_len * bs, embed_dim) @ wq.T).reshape(seq_len, bs, embed_dim),
    )

    q = q.reshape(seq_len, num_heads * bs, head_dim).transpose(0, 1)
    k = k.reshape(seq_len, num_heads * bs, head_dim).transpose(0, 1)
    v = v.reshape(seq_len, num_heads * bs, head_dim).transpose(0, 1)

    scale = 1.0 / head_dim**0.5

    # This is equivalent to the s=torch.bmm(q, k.transpose(1, 2))
    s = torch.zeros(bs * num_heads, seq_len, seq_len)
    for i in range(num_heads * bs):
        s[i] = q[i] @ k[i].T

    s = torch.softmax(s * scale, dim=-1)

    s_reshaped = s.reshape(bs, num_heads, seq_len, seq_len)

    s_avr = s_reshaped.sum(1) / num_heads  # if average_attn_weights is True
    # s_avr is what is returned by the function, but the it uses the full s

    torch.allclose(attn_output_weights, s_avr)
    y = torch.bmm(s, v)
    y = y.transpose(0, 1).reshape(seq_len * bs, embed_dim)
    y = y @ mha.out_proj.weight.T
    y = y.reshape(seq_len, bs, embed_dim)

    torch.allclose(attn_output, y)


@torch.inference_mode()
def test_reference_attention_n_heads_4_and_batch_first():
    embed_dim = 128
    num_heads = 8
    seq_len = 10
    bs = 4
    head_dim = embed_dim // num_heads

    # These are the default values, but with kdim = vdim = embed_dim
    mha = nn.MultiheadAttention(
        embed_dim=embed_dim,
        num_heads=num_heads,
        dropout=0,
        bias=False,
        add_bias_kv=False,
        add_zero_attn=False,
        batch_first=True,
    )

    assert mha.in_proj_weight.shape == torch.Size([embed_dim * 3, embed_dim])
    assert mha.out_proj.weight.shape == torch.Size([embed_dim, embed_dim])

    x = torch.rand(bs, seq_len, embed_dim)

    attn_output, attn_output_weights = mha(x, x, x)

    assert attn_output.shape == torch.Size([bs, seq_len, embed_dim])
    assert attn_output_weights.shape == torch.Size([bs, seq_len, seq_len])

    wq, wk, wv = torch.split(
        mha.in_proj_weight, [embed_dim, embed_dim, embed_dim], dim=0
    )

    q = torch.matmul(x, wq.T)
    k = torch.matmul(x, wk.T)
    v = torch.matmul(x, wv.T)

    assert k.shape == torch.Size([bs, seq_len, embed_dim])

    # It is not necessary to reshape the input, this is already
    # how matmul works for higher dimensions
    torch.allclose(
        x @ wq.T,
        (x.reshape(seq_len * bs, embed_dim) @ wq.T).reshape(bs, seq_len, embed_dim),
    )

    q = q.reshape(num_heads * bs, seq_len, head_dim)
    k = k.reshape(num_heads * bs, seq_len, head_dim)
    v = v.reshape(num_heads * bs, seq_len, head_dim)

    scale = 1.0 / head_dim**0.5

    s_slow = torch.zeros(bs * num_heads, seq_len, seq_len)
    for i in range(num_heads * bs):
        s_slow[i] = q[i] @ k[i].T

    # functionally equivalent to the below.
    s = torch.bmm(q, k.transpose(1, 2))
    assert torch.allclose(s, s_slow)

    s = torch.softmax(s * scale, dim=-1)

    s_reshaped = s.reshape(bs, num_heads, seq_len, seq_len)

    s_avr = s_reshaped.sum(1) / num_heads  # if average_attn_weights is True
    # s_avr is what is returned by the function, but the it uses the full s

    torch.allclose(attn_output_weights, s_avr)
    y = torch.bmm(s, v)
    y = y.transpose(0, 1).reshape(seq_len * bs, embed_dim)
    y = y @ mha.out_proj.weight.T
    y = y.reshape(bs, seq_len, embed_dim)

    torch.allclose(attn_output, y)


@torch.inference_mode()
def test_reference_attention_batch_first():
    embed_dim = 128
    num_heads = 8
    seq_len = 10
    bs = 4
    head_dim = embed_dim // num_heads
    print("head_dim:", head_dim)

    # These are the default values, but with kdim = vdim = embed_dim
    mha = nn.MultiheadAttention(
        embed_dim=embed_dim,
        num_heads=num_heads,
        dropout=0,
        bias=False,
        add_bias_kv=False,
        add_zero_attn=False,
        batch_first=True,
    )

    assert mha.in_proj_weight.shape == torch.Size([embed_dim * 3, embed_dim])
    assert mha.out_proj.weight.shape == torch.Size([embed_dim, embed_dim])

    x = torch.rand(bs, seq_len, embed_dim)
    attn_output, attn_output_weights = mha(x, x, x)

    assert attn_output.shape == torch.Size([bs, seq_len, embed_dim])
    assert attn_output_weights.shape == torch.Size([bs, seq_len, seq_len])

    # all of which are of shape [embed_dim, embed_dim]
    # my implementation is like [n_heads embed_dim d_head]
    # (gets einoped out anyway)
    wq, wk, wv = torch.split(
        mha.in_proj_weight, [embed_dim, embed_dim, embed_dim], dim=0
    )

    assert wq.shape == torch.Size([embed_dim, embed_dim])  # same for wk, wv

    q = torch.matmul(x, wq.T)
    k = torch.matmul(x, wk.T)
    v = torch.matmul(x, wv.T)

    # same shape for Q, V.
    assert k.shape == torch.Size([bs, seq_len, embed_dim])
    # in my implementation it's  [bs, seq_len, n_head, d_head] (n_head*d_head=embed_dim)

    # It is not necessary to reshape the input, this is already
    # how matmul works for higher dimensions
    torch.allclose(
        x @ wq.T,
        (x.reshape(seq_len * bs, embed_dim) @ wq.T).reshape(bs, seq_len, embed_dim),
    )

    q = q.reshape(num_heads * bs, seq_len, head_dim)
    k = k.reshape(num_heads * bs, seq_len, head_dim)
    v = v.reshape(num_heads * bs, seq_len, head_dim)

    scale = 1.0 / head_dim**0.5

    s = torch.bmm(q, k.transpose(1, 2))
    s = torch.softmax(s * scale, dim=-1)

    # if average_attn_weights is True,
    # s_avr is what is returned by the function, but the it uses the full score.
    s_avr = s.reshape(bs, num_heads, seq_len, seq_len).sum(1) / num_heads
    torch.allclose(attn_output_weights, s_avr)

    y: Float[Tensor, "bs*n_heads seq_len d_head"]
    y = torch.bmm(s, v)
    y: Float[Tensor, "bs*seq_len n_heads*d_head"]
    y = y.transpose(0, 1).reshape(seq_len * bs, embed_dim)
    print("y", y.shape)
    # [embed_dim, embed_dim] @ [embed_dim, embed_dim]
    y = y @ mha.out_proj.weight.T  # out_proj is of shape [embed_dim, embed_dim]
    y = y.reshape(bs, seq_len, embed_dim)

    torch.allclose(attn_output, y)
    # xxx


@torch.inference_mode()
def test_attention_equivalence_no_bias():
    # Set up parameters
    batch_size = 4
    seq_len = 10
    embed_dim = 128
    num_heads = 8
    head_dim = embed_dim // num_heads

    # Initialize your custom Attention
    custom_attn = Attention(
        d_head=head_dim,
        num_heads=num_heads,
        embed_dim=embed_dim,
        dropout=0.0,
        bias=False,
        add_bias_kv=False,
        add_zero_attn=False,
        batch_first=True,
    )

    # Initialize nn.MultiheadAttention
    mha = nn.MultiheadAttention(
        embed_dim=embed_dim,
        num_heads=num_heads,
        dropout=0.0,
        bias=False,
        add_bias_kv=False,
        add_zero_attn=False,
        batch_first=True,
    )

    # Set the weights of nn.MultiheadAttention to match your custom Attention
    with torch.no_grad():
        # Set in_proj_weight
        mha.in_proj_weight.copy_(
            torch.cat(
                [
                    custom_attn.W_Q.transpose(1, 2).reshape(embed_dim, embed_dim),
                    custom_attn.W_K.transpose(1, 2).reshape(embed_dim, embed_dim),
                    custom_attn.W_V.transpose(1, 2).reshape(embed_dim, embed_dim),
                ]
            )
        )

        # Set out_proj weight
        mha.out_proj.weight.copy_(custom_attn.W_O.reshape(embed_dim, embed_dim).T)

    # Generate random input
    x = torch.rand(batch_size, seq_len, embed_dim)

    # Run both attention mechanisms
    custom_output, custom_attn_weights = custom_attn(x)
    mha_output, mha_attn_weights = mha(x, x, x)

    # Check output shapes
    assert (
        custom_output.shape
        == mha_output.shape
        == torch.Size([batch_size, seq_len, embed_dim])
    )
    assert (
        custom_attn_weights.shape
        == mha_attn_weights.shape
        == torch.Size([batch_size, seq_len, seq_len])
    )

    # print(custom_output)
    # print(mha_output)

    log.info("x.shape", x=x.shape)

    log.info(
        "attention weights",
        mine=custom_attn_weights.shape,
        torch=mha_attn_weights.shape,
    )
    # mine: [4, 8, 10, 10]
    # torch [4, 10, 11]

    # Compare outputs
    max_diff_weights = torch.max(torch.abs(custom_attn_weights - mha_attn_weights))
    if max_diff_weights > 1e-4:
        print(
            f"Maximum difference between attention weights: {max_diff_weights.item()}"
        )
    assert torch.allclose(custom_attn_weights, mha_attn_weights, rtol=1e-4, atol=1e-4)

    # Check for close values (allowing for small numerical differences)
    max_diff_output = torch.max(torch.abs(custom_output - mha_output))
    print(f"Maximum difference between outputs: {max_diff_output.item()}")
    assert torch.allclose(custom_output, mha_output, rtol=1e-4, atol=1e-4)

    print(
        f"Test passed for batch_size={batch_size}, seq_len={seq_len}, embed_dim={embed_dim}, num_heads={num_heads}"
    )
