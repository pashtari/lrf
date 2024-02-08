import math
import random
from contextlib import contextmanager

import torch
from torch import nn
from torch.nn.modules.utils import _pair
import torch.nn.functional as F
from torchvision.models import resnet50


@contextmanager
def null_context():
    yield


class ResNet(nn.Module):
    def __init__(
        self,
        channels=3,
        num_classes=1000,
        compression_ratio=1,
        rescale=False,
        image_size=(224, 224),
        no_grad=True,
    ):
        super(ResNet, self).__init__()

        self.channels = channels
        self.num_classes = num_classes
        self.compression_ratio = _pair(compression_ratio)
        self.image_size = _pair(image_size)
        self.rescale = rescale
        self.no_grad = no_grad
        self.model = resnet50()

    def context(self):
        if self.no_grad:
            context = torch.no_grad()
        else:
            context = null_context()

        return context

    def compress(self, x):
        if self.compression_ratio[0] == self.compression_ratio[1]:
            cr = self.compression_ratio[0]
        else:
            cr = random.uniform(*self.compression_ratio)

        if cr != 1:
            scale_factor = 1 / math.sqrt(cr)
            x = F.interpolate(x, scale_factor=scale_factor)
            if self.rescale:
                x = F.interpolate(x, size=self.image_size)

        return x

    def forward(self, x):
        with self.context():
            z = self.compress(x)

        y = self.model(z)
        return y
