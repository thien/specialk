import torch
from torch import Tensor
from typing import Union, Tuple, Optional
import einops
from jaxtyping import Float


Pair = Tuple[int, int]
Triple = Tuple[int, int, int]
IntOrPair = Union[int, Pair]
IntOrTriple = Union[int, Triple]


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
    b, in_c, h, w, l = x.shape
    shape_out = (b, in_c, top + h + bottom, left + w + right, back + l + front)
    out = x.new_full(shape_out, pad_value)
    out[:, :, top : top + h, left : left + w, back : back + l] = x
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
    )  # new tensor with additional dimension for out_height, output_width, output_length.

    # return out @ weights
    return einops.einsum(
        out,
        weights,
        "batch in_c out_h kernel_h out_w kernel_w out_l kernel_l, out_c in_c kernel_h kernel_w kernel_l -> batch out_c out_h out_w out_l",
    )


def maxpool2d(
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

    x = pad2d(x, pad_w, pad_w, pad_h, pad_h, -t.inf)

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

    x = pad3d(x, pad_w, pad_w, pad_h, pad_h, pad_l, pad_l, -torch.inf)

    # get dimensions out.
    batch, in_channel, height, width, length = x.shape

    out_height = int(1 + ((height - ker_h) / str_h))
    out_width = int(1 + ((width - ker_w) / str_w))
    out_length = int(1 + ((length - ker_l) / str_l))

    a, b, c, d, e = x.stride()

    out = x.as_strided(
        size=(
            batch,
            in_channel,
            out_height,
            ker_h,
            out_width,
            ker_w,
            out_length,
            ker_l,
        ),
        stride=(a, b, c * str_h, c, d * str_w, d, e * str_l, e),
    )
    return torch.amax(out, dim=(3, 5, 7), keepdim=True).view(
        (batch, in_channel, out_height, out_width, out_length)
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