import math
from torch.optim.lr_scheduler import LRScheduler


class WarmupCosineAnnealingLR(LRScheduler):
    def __init__(self, optimizer, warmup_steps, total_steps, lr_min=0, last_epoch=-1):
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.lr_min = lr_min
        super(WarmupCosineAnnealingLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.last_epoch < self.warmup_steps:
            # Linear warmup
            slope = self.last_epoch / self.warmup_steps
            return [
                self.lr_min + slope * (base_lr - self.lr_min) for base_lr in self.base_lrs
            ]
        else:
            # Cosine annealing
            completed_steps = self.last_epoch - self.warmup_steps
            annealing_steps = self.total_steps - self.warmup_steps
            cosine_term = (1 + math.cos(math.pi * completed_steps / annealing_steps)) / 2
            return [
                self.lr_min + cosine_term * (base_lr - self.lr_min)
                for base_lr in self.base_lrs
            ]
