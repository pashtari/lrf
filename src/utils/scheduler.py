from torch.optim.lr_scheduler import LRScheduler
from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR, SequentialLR


class WarmupCosineAnnealingLR(LRScheduler):
    def __init__(
        self,
        optimizer,
        warmup_epochs,
        max_epochs,
        lr_min=0,
        warmup_decay=0.1,
        last_epoch=-1,
    ):
        self.warmup_epochs = warmup_epochs
        self.max_epochs = max_epochs
        self.lr_min = lr_min
        self.warmup_decay = warmup_decay
        self.optimizer = optimizer

        # Initialize warm-up scheduler
        warmup_scheduler = LinearLR(
            optimizer, start_factor=warmup_decay, total_iters=warmup_epochs
        )

        # Initialize cosine annealing scheduler
        cosine_scheduler = CosineAnnealingLR(
            optimizer, T_max=max_epochs - warmup_epochs, eta_min=lr_min
        )

        # Combine both schedulers using SequentialLR
        self.sequential_scheduler = SequentialLR(
            optimizer,
            schedulers=[warmup_scheduler, cosine_scheduler],
            milestones=[warmup_epochs],
            last_epoch=last_epoch,
        )

        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        return self.sequential_scheduler.get_lr()

    def step(self, epoch=None):
        self.sequential_scheduler.step()
