from tqdm import tqdm
import torch
import einops
import math
from specialk.core.utils import log

batch_size = 128
sequence_length = 100
filter_size = 5
conv_dim_output = 96

pooling_window_size = sequence_length - filter_size - 1



maxpool = torch.nn.MaxPool3d(kernel_size=(1, pooling_window_size, 1), stride=(1, 1, 1))

log.info("torch_maxpool", maxpool=maxpool)


def normal_maxpool_3d(ims):
    return maxpool(ims)


def pool_output(
    input: int, kernel: int, padding: int = 1, dilation: int = 1, stride: int = 1
) -> int:
    """
    you can use this formula [(W−K+2P)/S]+1.

    W is the input volume - in your case 128
    K is the Kernel size - in your case 5
    P is the padding - in your case 0 i believe
    S is the stride - which you have not provided.
    So, we input into the formula:

    Output_Shape = (128-5+0)/1+1
    """

    log.info(
        "pool input equation",
        input=input,
        kernel=kernel,
        padding=padding,
        dilation=dilation,
        stride=stride,
    )
    top = input + 2 * padding - (dilation * (kernel - 1)) - 1
    bottom = stride
    out = math.floor((top / bottom) + 1)
    log.info("pool output dim", out=out)
    return out


# this is the target output shape
pool_out_dim = pool_output(
    input=conv_dim_output, kernel=pooling_window_size, stride=1, padding=0
)


def einsum_maxpool_3d(ims):
    kernel = int(pooling_window_size / pool_out_dim)
    kernel = 32 if sequence_length == 100 else None
    return einops.reduce(
        ims, "batch seq (emb kernel) c -> batch seq emb c", "max", kernel=kernel
    )


x = torch.rand(batch_size, sequence_length, conv_dim_output, 1)

log.info("input", x=x.shape)

y = normal_maxpool_3d(x)

out_shape = torch.Size([batch_size, sequence_length, pool_out_dim, 1])


assert out_shape == y.shape, f"{y.shape} != out_shape={out_shape}"
log.info("Reference MaxPool3d", y=y.shape, y_ref=out_shape)

m = torch.nn.MaxPool2d(kernel_size=(pooling_window_size, 1), stride=(1, 1))

output = m(x)

log.info("maxpool2d", shape=output.shape)

assert torch.equal(y, output)