from typing import Any, Dict

import torch
from torchvision.transforms.v2 import Transform

import lrf


class JPEGTransform(Transform):
    def __init__(self, **kwargs):
        super().__init__()
        self.kwargs = kwargs

    def _transform(self, inpt: Any, params: Dict[str, Any]) -> Any:
        enocoded = lrf.pil_encode(inpt, format="JPEG", **self.kwargs)
        reconstructed = lrf.pil_decode(enocoded)
        bpp = lrf.get_bbp(inpt.shape[-2:], enocoded)
        return reconstructed, torch.tensor(bpp)


class SVDTransform(Transform):
    def __init__(self, **kwargs):
        super().__init__()
        self.kwargs = kwargs

    def _transform(self, inpt: Any, params: Dict[str, Any]) -> Any:
        enocoded = lrf.svd_encode(inpt, **self.kwargs)
        reconstructed = lrf.svd_decode(enocoded)
        bpp = lrf.get_bbp(inpt.shape[-2:], enocoded)
        return reconstructed, torch.tensor(bpp)


class IMFTransform(Transform):
    def __init__(self, **kwargs):
        super().__init__()
        self.kwargs = kwargs

    def _transform(self, inpt: Any, params: Dict[str, Any]) -> Any:
        enocoded = lrf.imf_encode(inpt, **self.kwargs)
        reconstructed = lrf.imf_decode(enocoded)
        bpp = lrf.get_bbp(inpt.shape[-2:], enocoded)
        return reconstructed, torch.tensor(bpp)
