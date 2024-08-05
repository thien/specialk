import torch

from specialk.models.ops import FuncMaxPool3d


def test_maxpool2d(n_tests=20):
    import numpy as np

    for i in range(n_tests):
        b = np.random.randint(1, 10)
        h = np.random.randint(10, 50)
        w = np.random.randint(10, 50)
        l = np.random.randint(10, 50)
        ci = np.random.randint(1, 20)
        stride = (
            None
            if np.random.random() < 0.5
            else tuple(np.random.randint(1, 5, size=(3,)))
        )
        kernel_size = tuple(np.random.randint(1, 10, size=(3,)))
        kH, kW, kL = kernel_size
        padding = (
            np.random.randint(0, 1 + kH // 2),
            np.random.randint(0, 1 + kW // 2),
            np.random.randint(0, 1 + kL // 2),
        )
        x = torch.randn((b, ci, h, w, l))
        my_output = FuncMaxPool3d(
            x,
            kernel_size,
            stride=stride,
            padding=padding,
        )
        torch_output = torch.max_pool3d(
            x,
            kernel_size,
            stride=stride,  # type: ignore (None actually is allowed)
            padding=padding,
        )
        torch.testing.assert_close(my_output, torch_output)
    print("All tests in `test_maxpool2d` passed!")
