from typing import Sequence

import torch
from ignite.metrics import Accuracy


class BPP(Accuracy):
    def __init__(self):
        super().__init__()
        self._sum_bpp = 0

    def update(self, output: Sequence[torch.Tensor]) -> None:
        if hasattr(output[0], "bpp"):
            bpps = output[0].bpp

            self._sum_bpp += torch.sum(bpps).to(self._device)
            self._num_examples += bpps.shape[0]

    def compute(self) -> float:
        ave = self._sum_bpp / self._num_examples if self._num_examples != 0 else -1
        return ave
