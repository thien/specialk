from typing import List

import numpy as np
from schedulefree import AdamWScheduleFree
from torch.optim import AdamW, Optimizer
from torch.optim.lr_scheduler import LambdaLR, _LRScheduler


class CosineWarmupScheduler(_LRScheduler):
    def __init__(self, optimizer: Optimizer, n_warmup_steps: int, max_iters: int):
        self.warmup = n_warmup_steps
        self.max_num_iters = max_iters
        super().__init__(optimizer)

    def get_lr(self) -> List[float]:
        lr_factor = self.get_lr_factor(epoch=self.last_epoch)
        return [base_lr * lr_factor for base_lr in self.base_lrs]

    def get_lr_factor(self, epoch: int) -> float:
        lr_factor = 0.5 * (1 + np.cos(np.pi * epoch / self.max_num_iters))
        if epoch <= self.warmup:
            lr_factor *= epoch * 1.0 / self.warmup
        return lr_factor
