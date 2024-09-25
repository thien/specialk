import pytest
import torch
from torch import LongTensor

from specialk.core.constants import EOS, PAD
from specialk.models.ops.ops import mask_out_special_tokens, n_tokens_correct


@pytest.mark.lightweight
def test_n_tokens_correct_2d():
    # Test Case 1: Basic test case with no padding
    pred1 = torch.tensor([[0.1, 0.9], [0.8, 0.2], [0.4, 0.6]])
    gold1 = torch.tensor([1, 0, 1])
    pad_token1 = -1
    assert n_tokens_correct(pred1, gold1, pad_token1) == 3

    # Test Case 2: 2 predictions are correct.
    pred2 = torch.tensor([[0.1, 0.9], [0.8, 0.2], [0.4, 0.6]])
    gold2 = torch.tensor([1, 0, 0])
    pad_token2 = -1
    assert n_tokens_correct(pred2, gold2, pad_token2) == 2

    # Test Case 3: All predictions are wrong
    pred3 = torch.tensor([[0.9, 0.1], [0.2, 0.8], [0.6, 0.4]])
    gold3 = torch.tensor([1, 0, 1])
    pad_token3 = -1
    assert n_tokens_correct(pred3, gold3, pad_token3) == 0

    # Test Case 4: Including padding tokens
    pred4 = torch.tensor([[0.1, 0.9], [0.8, 0.2], [0.4, 0.6], [0.7, 0.3]])
    gold4 = torch.tensor([1, 0, -1, 1])
    pad_token4 = -1
    assert n_tokens_correct(pred4, gold4, pad_token4) == 2

    # Test Case 5: All tokens are padding
    pred5 = torch.tensor([[0.1, 0.9], [0.8, 0.2], [0.4, 0.6]])
    gold5 = torch.tensor([-1, -1, -1])
    pad_token5 = -1
    assert n_tokens_correct(pred5, gold5, pad_token5) == 0


@pytest.mark.lightweight
def test_n_tokens_correct_3d():
    # Test Case 1: Basic test case with no padding
    pred1 = torch.tensor([[[0.1, 0.9], [0.8, 0.2], [0.4, 0.6]]])
    gold1 = torch.tensor([[1, 0, 1]])
    pad_token1 = -1
    assert n_tokens_correct(pred1, gold1, pad_token1) == 3

    # Test Case 2: 2 predictions are correct
    pred2 = torch.tensor([[[0.1, 0.9], [0.8, 0.2], [0.4, 0.6]]])
    gold2 = torch.tensor([[1, 0, 0]])
    pad_token2 = -1
    assert n_tokens_correct(pred2, gold2, pad_token2) == 2

    # Test Case 3: All predictions are wrong
    pred3 = torch.tensor([[[0.9, 0.1], [0.2, 0.8], [0.6, 0.4]]])
    gold3 = torch.tensor([[1, 0, 1]])
    pad_token3 = -1
    assert n_tokens_correct(pred3, gold3, pad_token3) == 0

    # Test Case 4: Including padding tokens
    pred4 = torch.tensor([[[0.1, 0.9], [0.8, 0.2], [0.4, 0.6], [0.7, 0.3]]])
    gold4 = torch.tensor([[1, 0, -1, 1]])
    pad_token4 = -1
    assert n_tokens_correct(pred4, gold4, pad_token4) == 2

    # Test Case 5: All tokens are padding
    pred5 = torch.tensor([[[0.1, 0.9], [0.8, 0.2], [0.4, 0.6]]])
    gold5 = torch.tensor([[-1, -1, -1]])
    pad_token5 = -1
    assert n_tokens_correct(pred5, gold5, pad_token5) == 0

    # Test Case 6: Batch processing with padding
    pred6 = torch.tensor(
        [
            [
                [0.1, 0.9],  # 1 (correct)
                [0.8, 0.2],  # 0 (correct)
                [0.4, 0.6],  # 1 skip because we'll pad
            ],
            [
                [0.7, 0.3],  # 0 (we get this wrong)
                [0.2, 0.8],  # 1 (we'll get this wrong)
                [0.6, 0.4],  # 0 (we'll get this wrong)
            ],
        ]
    )  # we should only get 2 correct items.
    gold6 = torch.tensor([[1, 0, -1], [1, 0, 1]])
    pad_token6 = -1
    assert n_tokens_correct(pred6, gold6, pad_token6) == 2


@pytest.mark.lightweight
def test_mask_out_special_tokens():
    """Mostly for my sanity checking to see if masking is actually happening."""
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
