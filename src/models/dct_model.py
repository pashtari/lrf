from typing import Sequence, Iterable

import random
import torch
from torch import nn
from torch.nn.modules.utils import _pair
from torchvision.models import resnet50

from .compression import DCT
from ..utils.helper import null_context


class DCTModel(nn.Module):
    def __init__(
        self,
        net=None,
        pad=False,
        original_size=224,
        new_size=None,
        no_grad=True,
        domain="compressed",
        **kwargs
    ):
        super(DCTModel, self).__init__()

        self.updated_compression_ratio = None
        self.original_size = _pair(original_size)
        if new_size is None:
            self.new_size = (self.original_size,)
        elif isinstance(new_size, int):
            self.new_size = ((new_size, new_size),)
        elif isinstance(new_size, (Sequence, Iterable)) and not isinstance(new_size, str):
            self.new_size = tuple(_pair(s) for s in new_size)
        else:
            raise ValueError("`new_size` type is incorrect.")

        self.pad = pad
        assert domain in {"compressed", "decompressed", "com", "dec"}
        self.domain = domain
        self.no_grad = no_grad
        if net is None:
            net = resnet50

        self.net = net(**kwargs)
        self.dct = DCT()

    def context(self):
        if self.no_grad:
            context = torch.no_grad()
        else:
            context = null_context()

        return context

    def transform(self, x):
        with self.context():
            new_size = random.choice(self.new_size)
            compression_ratio = (self.original_size[0] * self.original_size[1]) / (
                new_size[0] * new_size[1]
            )
            if self.domain in {"com", "compressed"}:
                z = self.dct.compress(x, compression_ratio=compression_ratio)
            elif self.domain in {"dec", "decompressed"}:
                if compression_ratio == 1:
                    z = x
                else:
                    z = self.dct(x, compression_ratio=compression_ratio, pad=self.pad)

        return z

    def forward(self, x):
        z = self.transform(x)
        y = self.net(z)
        return y
