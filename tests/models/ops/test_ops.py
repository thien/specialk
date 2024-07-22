from specialk.models.ops import mask_out_special_tokens
import torch
from torch import LongTensor
from specialk.core.constants import PAD, EOS


def test_mask_out_special_tokens():
    test_cases = [
        (
            LongTensor([[20, 44, 55, 245, 62626, EOS, 222, 222, 222]]),
            LongTensor([[20, 44, 55, 245, 62626, EOS, PAD, PAD, PAD]]),
        ),
        (
            LongTensor([[20, 44, 55, 245, 62626, 222, 222, 222, EOS]]),
            LongTensor([[20, 44, 55, 245, 62626, 222, 222, 222, EOS]]),
        ),
        (
            LongTensor([[20, 44, 55, 245, 62626, 222, 222, 222, 222]]),
            LongTensor([[20, 44, 55, 245, 62626, 222, 222, 222, 222]]),
        ),
        (
            LongTensor([[EOS, 222, 222, 222, 222, 222, 222, 222]]),
            LongTensor([[EOS, PAD, PAD, PAD, PAD, PAD, PAD, PAD]]),
        ),
        (  # combines all of the above.
            LongTensor(
                [
                    [20, 44, 55, 245, 62626, EOS, 222, 222, 222],
                    [20, 44, 55, 245, 62626, 222, 222, 222, EOS],
                    [20, 44, 55, 245, 62626, 222, 222, 222, 222],
                    [EOS, 222, 222, 222, 222, 222, 222, 222, 22],
                ]
            ),
            LongTensor(
                [
                    [20, 44, 55, 245, 62626, EOS, PAD, PAD, PAD],
                    [20, 44, 55, 245, 62626, 222, 222, 222, EOS],
                    [20, 44, 55, 245, 62626, 222, 222, 222, 222],
                    [EOS, PAD, PAD, PAD, PAD, PAD, PAD, PAD, PAD],
                ]
            ),
        ),
    ]
    devices = ["cpu"]
    if torch.backends.mps.is_available():
        devices.append("mps")
    if torch.cuda.is_available():
        devices.append("cuda")

    for device in devices:
        for x, y in test_cases:
            x = x.to(device)
            y = y.to(device)
            y_hat = mask_out_special_tokens(x, EOS, PAD)
            assert torch.equal(y, y_hat)
