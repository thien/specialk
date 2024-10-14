from typing import Optional, Tuple, Union

import einops
import torch
from jaxtyping import Float, Int
from torch import Tensor

from specialk import Constants

Pair = Tuple[int, int]
Triple = Tuple[int, int, int]
IntOrPair = Union[int, Pair]
IntOrTriple = Union[int, Triple]


def token_accuracy(pred: Tensor, gold: Tensor, pad_token: int) -> float:
    accuracy = n_tokens_correct(pred, gold, pad_token)
    total = gold.ne(pad_token).sum()
    return accuracy / total


def n_tokens_correct(
    pred: Tensor, gold: Tensor, pad_token: Optional[int] = None
) -> int:
    """
    Calculate the number of correct tokens.
    This ignores padding tokens; This is not differentible.

    Args:
        pred (Tensor): The predicted tensor of shape (N, C) where N
        is the number of tokens and C is the number of classes.
        gold (Tensor): The ground truth tensor of shape (N,) containing
        the correct classes for each token.
        pad_token (int): The token index used for padding.

    Returns:
        int: The number of correct tokens, excluding padding tokens.
    """
    pred_indices: Int[Tensor, "n"] = pred.argmax(dim=-1).view(-1)
    gold_flat: Int[Tensor, "n"] = gold.view(-1)

    n_correct = pred_indices.eq(gold_flat)
    if pad_token is not None:
        non_pad_mask = gold_flat.ne(pad_token)
        n_correct = n_correct.masked_select(non_pad_mask)
    return int(n_correct.sum().item())


def force_pair(v: IntOrPair) -> Pair:
    """Convert v to a pair of int, if it isn't already."""
    if isinstance(v, tuple):
        if len(v) != 2:
            raise ValueError(v)
        return (int(v[0]), int(v[1]))
    elif isinstance(v, int):
        return (v, v)
    raise ValueError(v)


def force_triple(v: IntOrTriple) -> Triple:
    """Convert v to a triple of int, if it isn't already."""
    if isinstance(v, tuple):
        if len(v) != 3:
            raise ValueError(v)
        return (int(v[0]), int(v[1]), int(v[2]))
    elif isinstance(v, int):
        return (v, v, v)
    raise ValueError(v)


def pad2d(
    x: Tensor, left: int, right: int, top: int, bottom: int, pad_value: float
) -> Tensor:
    """Return a new tensor with padding applied to the edges.

    x: shape (batch, in_channels, height, width), dtype float32

    Return: shape (batch, in_channels, top + height + bottom, left + width + right)
    """
    b, in_c, h, w = x.shape
    shape_out = (b, in_c, top + h + bottom, left + w + right)
    out = x.new_full(shape_out, pad_value)
    out[:, :, top : top + h, left : left + w] = x
    return out


def pad3d(
    x: Tensor,
    left: int,
    right: int,
    top: int,
    bottom: int,
    front: int,
    back: int,
    pad_value: float,
) -> Tensor:
    """Return a new tensor with padding applied to the edges.

    x: shape (batch, in_channels, height, width, length), dtype float32

    Return: shape (batch, in_channels, top + height + bottom, left + width + right, back + length + front)
    """
    b, in_c, h, w, length = x.shape
    shape_out = (b, in_c, top + h + bottom, left + w + right, back + length + front)
    out = x.new_full(shape_out, pad_value)
    out[:, :, top : top + h, left : left + w, back : back + length] = x
    return out


def conv3d(
    x: Float[Tensor, "b ic h w"],
    weights: Float[Tensor, "oc ic kh kw"],
    stride: IntOrPair = 1,
    padding: IntOrPair = 0,
) -> Float[Tensor, "b oc oh ow"]:
    """
    Like torch's conv2d using bias=False

    x: shape (batch, in_channels, height, width, length)
    weights: shape (out_channels, in_channels, kernel_height, kernel_width, kernel_length)

    Returns: shape (batch, out_channels, output_height, output_width, output_length)
    """

    str_h, str_w, str_l = force_triple(stride)
    pad_h, pad_w, pad_l = force_triple(padding)

    x = pad3d(x, pad_w, pad_w, pad_h, pad_h, pad_l, pad_l, 0)

    # get dimensions out.
    batch, in_channel_src, height, width, length = x.shape
    _, in_channels, kernel_height, kernel_width, kernel_length = weights.shape
    assert in_channels == in_channel_src

    out_height = int(1 + ((height - kernel_height) / str_h))
    out_width = int(1 + ((width - kernel_width) / str_w))
    out_length = int(1 + ((length - kernel_length) / str_l))

    a, b, c, d, e = x.stride()

    out = x.as_strided(
        size=(
            batch,
            in_channels,
            out_height,
            kernel_height,
            out_width,
            kernel_width,
            out_length,
            kernel_length,
        ),
        stride=(a, b, c * str_h, c, d * str_w, d, d * str_w, e, e * str_l),
    )  # new tensor with additional dimension for out_height,
    # output_width, output_length.

    # return out @ weights
    out_shape = "batch in_c out_h kernel_h out_w kernel_w out_l kernel_l"
    weights_shape = "out_c in_c kernel_h kernel_w kernel_l"
    return einops.einsum(
        out,
        weights,
        f"{out_shape}, {weights_shape} -> batch out_c out_h out_w out_l",
    )


def FuncMaxPool2d(
    x: Float[Tensor, "b ic h w"],
    kernel_size: IntOrPair,
    stride: Optional[IntOrPair] = None,
    padding: IntOrPair = 0,
) -> Float[Tensor, "b ic oh ow"]:
    """
    Like PyTorch's maxpool2d.

    x: shape (batch, channels, height, width)
    stride: if None, should be equal to the kernel size

    Return: (batch, channels, output_height, output_width)
    """

    stride = stride if stride else kernel_size
    str_h, str_w = force_pair(stride)
    pad_h, pad_w = force_pair(padding)
    ker_h, ker_w = force_pair(kernel_size)

    x = pad2d(x, pad_w, pad_w, pad_h, pad_h, -torch.inf)

    batch, in_channel, height, width = x.shape

    out_height = int(1 + ((height - ker_h) / str_h))
    out_width = int(1 + ((width - ker_w) / str_w))

    a, b, c, d = x.stride()

    out = x.as_strided(
        size=(batch, in_channel, out_height, ker_h, out_width, ker_w),
        stride=(a, b, c * str_h, c, d * str_w, d),
    )
    return torch.amax(out, dim=(3, 5), keepdim=True).view(
        (batch, in_channel, out_height, out_width)
    )


def FuncMaxPool3d(
    x: Float[Tensor, "b ic h w l"],
    kernel_size: IntOrTriple,
    stride: Optional[IntOrTriple] = None,
    padding: IntOrTriple = 0,
) -> Float[Tensor, "b ic oh ow ol"]:
    """
    Like PyTorch's maxpool3d.
    This exists because a native implemenation on mps doesn't exist yet,
    so we're making our own from scratch.

    x: shape (batch, channels, height, width, length)
    stride: if None, should be equal to the kernel size

    Return: (batch, channels, output_height, output_width, output_length)
    """

    stride = stride if stride else kernel_size
    str_h, str_w, str_l = force_triple(stride)
    pad_h, pad_w, pad_l = force_triple(padding)
    ker_h, ker_w, ker_l = force_triple(kernel_size)

    # x = pad3d(x, pad_w, pad_w, pad_h, pad_h, pad_l, pad_l, -torch.inf)
    pad = (pad_h, pad_h, pad_w, pad_w, pad_l, pad_l)
    # print(pad)
    x = torch.nn.functional.pad(x, pad)

    # get dimensions out.
    batch, in_channel, height, width, length = x.shape

    out_height = int(1 + ((height - ker_h) / str_h))
    out_width = int(1 + ((width - ker_w) / str_w))
    out_length = int(1 + ((length - ker_l) / str_l))

    a, b, c, d, e = x.stride()

    return torch.amax(
        x.as_strided(
            size=(
                batch,
                in_channel,
                out_height,
                out_width,
                out_length,
                ker_h,
                ker_w,
                ker_l,
            ),
            stride=(a, b, c * str_h, d * str_w, e * str_l, c, d, e),
        ),
        dim=(-1, -2, -3),
    )


class MaxPool3d(torch.nn.Module):
    def __init__(
        self,
        kernel_size: IntOrTriple,
        stride: Optional[IntOrTriple] = None,
        padding: IntOrTriple = 0,
    ):
        """
        Like PyTorch's maxpool3d.
        This exists because a native implemenation on mps doesn't exist yet,
        so we're making our own from scratch.

        x: shape (batch, channels, height, width, length)
        stride: if None, should be equal to the kernel size

        Return: (batch, channels, output_height, output_width, output_length)
        """
        super().__init__()
        stride = stride if stride else kernel_size
        self.stride = force_triple(stride)
        self.padding = force_triple(padding)
        self.kernel = force_triple(kernel_size)

    def forward(self, x: Tensor) -> Tensor:
        return FuncMaxPool3d(x, self.kernel, self.stride, self.padding)


def mask_out_special_tokens(
    x: Int[Tensor, "batch seq"],
    eos_index: int = Constants.EOS,
    pad_index: int = Constants.PAD,
) -> Int[Tensor, "batch seq"]:
    """
    Mask out tokens beyond the EOS token.

    Args:
        x (Tensor): Tensor generated by a NMT model.
        pad_token (int, Optional): index of PAD.
        eos_index (int, Optional): index of EOS.

    Returns:
        Tensor: Tensor with any tokens past the EOS replaced by PAD tokens.
    """
    n_batch, seq_len = x.shape
    mask = (x == eos_index).to(torch.long)

    # Get the positions of the first occurrence in each row
    pos_eos = mask.argmax(dim=-1)
    # If a row does not contain the EOS index,
    # then just say it's at the end of the sequence.
    pos_eos[~mask.any(dim=-1)] = seq_len

    mask = torch.arange(seq_len, device=x.device).expand(n_batch, seq_len).T
    mask = mask <= pos_eos
    mask = mask.T
    x_masked = x * mask

    if pad_index != Constants.PAD:
        x_masked[x == 0] = pad_index

    return x_masked
