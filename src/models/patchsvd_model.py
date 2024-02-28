from functools import partial
import math
import random
from typing import Iterable, Sequence
from einops import rearrange

import torch
from torch import nn
from torch.nn.modules.utils import _pair
from torchvision.models.feature_extraction import create_feature_extractor


from .compression import PatchSVD
from .resnetnd import resnet50
from ..utils.helper import null_context


class PatchSVDModel(nn.Module):
    def __init__(
        self,
        net=None,
        patch_size=8,
        rank=None,
        pad=False,
        no_grad=True,
        domain="compressed",
        **kwargs
    ):
        super().__init__()

        self.patch_size = _pair(patch_size)

        if rank is None:
            self.rank = (-1,)
        elif isinstance(rank, int):
            self.rank = (rank,)
        elif isinstance(rank, (Sequence, Iterable)) and not isinstance(rank, str):
            self.new_size = tuple(_pair(s) for s in rank)
        else:
            raise ValueError("`new_size` type is incorrect.")

        self.pad = pad
        self.no_grad = no_grad

        self.patch_svd = PatchSVD(patch_size=patch_size)
        assert domain in ("compressed", "decompressed", "com", "dec")
        self.domain = domain

        if domain in ("compressed", "com"):
            if net is None:
                resnet = resnet50(in_channels=64, spatial_dims=3, conv1_kernel_size=3,conv1_stride=1,no_maxpool=True,num_classes=1000,**kwargs) 
                self.net = nn.Sequential(nn.Linear(1, 64),create_feature_extractor(resnet, {"layer4": "features"}))
                self.fc = nn.Linear(2048, kwargs.get("num_classes", 1000))
            else:
                self.net = net(**kwargs)
        else:
            self.net = net(**kwargs)

    def context(self):
        if self.no_grad:
            context = torch.no_grad()
        else:
            context = null_context()

        return context

    def transform(self, x):
        with self.context():
            rank = random.choice(self.rank)
            compression_ratio = self.patch_svd.get_compression_ratio(x.shape[-2:], rank)
            if self.domain in {"com", "compressed"}:
                uv = self.patch_svd.compress(x, compression_ratio=compression_ratio)
                z = self.patch_svd.depatchify_uv(x, *uv)
            elif self.domain in {"dec", "decompressed"}:
                if compression_ratio == 1:
                    z = x
                else:
                    z = self.patch_svd(
                        x, compression_ratio=compression_ratio, pad=self.pad
                    )
        return z

    def forward(self, x):
        z = self.transform(x)

        if isinstance(z, torch.Tensor):
            y = self.net(z)
        else:
            patch_svd_modelu, v = z
            u = self.net(u)
            v = self.net(v)

            u = torch.flatten(u, 3)
            v = torch.flatten(v, 3)

            c = int(math.log2(u.shape[1]))//2
            u = rearrange(u, "b (c e) r -> b c e r", c = c)
            v = rearrange(v, "b (c e) r -> b c e r", c = c)

            y = torch.einsum("b c e r, b d e r -> b c d", u, v)

            y = self.fc(y)

        return y

