import torch


def check_torch_device() -> str:
    """Check which device is available. Returns one of {cpu, cuda, mps}"""
    if torch.backends.mps.is_available():
        return "mps"
    elif torch.cuda.is_available():
        return "cuda"
    return "cpu"
