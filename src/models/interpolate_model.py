from typing import Sequence

import random

import torch
from torch import nn
from torch.nn.modules.utils import _pair
from torchvision.models import resnet50

from .compression import Interpolate
from ..utils.helper import null_context


class InterpolateModel(nn.Module):
    def __init__(
        self,
        net=None,
        rescale=False,
        original_size=224,
        new_size=None,
        no_grad=True,
    ):
        super(InterpolateModel, self).__init__()

        self.updated_compression_ratio = None
        self.original_size = _pair(original_size)
        if new_size is None:
            self.new_size = (self.original_size,)
        elif new_size in ("all", "rand", "random"):
            self.new_size = self.get_all_feasible_sizes()
        elif isinstance(new_size, int):
            self.new_size = ((new_size, new_size),)
        elif isinstance(new_size, Sequence) and not isinstance(new_size, str):
            self.new_size = tuple(_pair(s) for s in new_size)
        else:
            raise ValueError("`new_size` type is incorrect.")

        self.rescale = rescale
        self.no_grad = no_grad
        if net is None:
            net = resnet50

        self.net = net()
        self.interpolate = Interpolate()

    def context(self):
        if self.no_grad:
            context = torch.no_grad()
        else:
            context = null_context()

        return context

    def get_all_feasible_sizes(self):
        heights = [16 * i for i in range(3, self.original_size[0] // 16)]
        widths = [16 * j for j in range(3, self.original_size[1] // 16)]
        all_feasible_sizes = [*zip(heights, widths)]
        return all_feasible_sizes

    def transform(self, x):
        with self.context():
            new_size = random.choice(self.new_size)
            compression_ratio = (self.original_size[0]*self.original_size[1]) // (new_size[0]*new_size[1])
            if compression_ratio == 1:
                z = x
            else:
                z = self.interpolate.compress(x, compression_ratio=compression_ratio)
            if self.rescale and compression_ratio != 1:
                z = self.interpolate.decompress(z, original_size=self.original_size)
        return z

    def forward(self, x):
        z = self.transform(x)
        y = self.net(z)
        return y
