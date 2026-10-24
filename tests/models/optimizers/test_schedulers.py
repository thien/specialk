import torch

from specialk.models.optimizers.schedulers import CosineWarmupScheduler


def test_cosine_warmup_scheduler_returns_python_floats():
    model = torch.nn.Linear(1, 1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.1)
    scheduler = CosineWarmupScheduler(optimizer, n_warmup_steps=2, max_iters=10)

    assert type(scheduler.get_lr_factor(epoch=1)) is float

    optimizer.step()
    scheduler.step()

    for param_group in optimizer.param_groups:
        # numpy scalars become float64 tensors in Lightning's LR monitor,
        # which fails on MPS.
        assert type(param_group["lr"]) is float
