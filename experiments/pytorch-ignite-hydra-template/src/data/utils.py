import functools
import torch
import lrf
from torchvision.transforms.v2 import Transform
from torch.utils.data import default_collate
from typing import Any, Dict, Sequence
from ignite.metrics import Accuracy


def collate_fn_wrapper(batched_transform):
    @functools.wraps(batched_transform)
    def collate_fn(batch):
        x, y = default_collate(batch)
        x, _ = batched_transform(x, y)
        return (x, y)

    return collate_fn


class Real_bpp(Accuracy):
    def __init__(self):
        super().__init__()
        self._sum_bpp = 0

    def update(self, output: Sequence[torch.Tensor]) -> None:
        if hasattr(output[0], "real_bpp"):
            real_bpps = output[0].real_bpp

            self._sum_bpp += torch.sum(real_bpps).to(self._device)
            self._num_examples += real_bpps.shape[0]

    def compute(self) -> float:
        ave = self._sum_bpp / self._num_examples if self._num_examples != 0 else -1
        return ave


class jpeg_transformer(Transform):
    def __init__(self, **kwargs):
        super().__init__()
        self.kwargs = kwargs

    def _transform(self, inpt: Any, params: Dict[str, Any]) -> Any:
        encoded = lrf.pil_encode(inpt, format="JPEG", **self.kwargs)
        reconstructed = lrf.pil_decode(encoded)
        real_bpp = lrf.get_bbp(inpt.shape[-2:], encoded)
        return reconstructed, torch.tensor(real_bpp)


class svd_transformer(Transform):
    def __init__(self, **kwargs):
        super().__init__()
        self.kwargs = kwargs

    def _transform(self, inpt: Any, params: Dict[str, Any]) -> Any:
        encoded = lrf.svd_encode(inpt, **self.kwargs)
        reconstructed = lrf.svd_decode(encoded)
        real_bpp = lrf.get_bbp(inpt.shape[-2:], encoded)
        return reconstructed, torch.tensor(real_bpp)


class imf_transformer(Transform):
    def __init__(self, **kwargs):
        super().__init__()
        self.kwargs = kwargs

    def _transform(self, inpt: Any, params: Dict[str, Any]) -> Any:
        encoded = lrf.imf_encode(inpt, **self.kwargs)
        reconstructed = lrf.imf_decode(encoded)
        real_bpp = lrf.get_bbp(inpt.shape[-2:], encoded)
        return reconstructed, torch.tensor(real_bpp)
