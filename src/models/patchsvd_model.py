from typing import Iterable, Sequence
import math
import random

import torch
from torch import nn
from torch.nn.modules.utils import _pair
from torchvision.models.feature_extraction import create_feature_extractor
import torchvision
from einops import rearrange

from .compression import PatchSVD
from .resnet import resnet50
from ..utils.helper import null_context


class PatchSVDModel(nn.Module):
    def __init__(
        self,
        net=None,
        num_classes=1000,
        patch_size=8,
        rank=None,
        domain="compressed",
        no_grad=True,
        **kwargs,
    ):
        super().__init__()

        self.num_classes = num_classes
        self.patch_size = _pair(patch_size)

        if rank is None:
            self.rank = (-1,)
        elif isinstance(rank, int):
            self.rank = (rank,)
        elif isinstance(rank, (Sequence, Iterable)) and not isinstance(rank, str):
            self.rank = tuple(_pair(s) for s in rank)
        else:
            raise ValueError("`new_size` type is incorrect.")

        self.real_compression_ratio = None
        self.patch_svd = PatchSVD(patch_size=self.patch_size)

        assert domain in ("compressed", "decompressed", "com", "dec")
        self.domain = domain

        self.no_grad = no_grad

        if domain in ("compressed", "com"):
            self.net = UVModel(
                backbone=net,
                u_channels=1,
                v_channels=3,
                num_classes=self.num_classes,
                **kwargs,
            )
        else:
            if net is None:
                self.net = torchvision.models.resnet50(
                    num_classes=self.num_classes, **kwargs
                )
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
            if self.domain in ("compressed", "com"):
                uv = self.patch_svd.compress(x, rank=rank)
                z = self.patch_svd.depatchify_uv(x, *uv)
                self.real_compression_ratio = self.patch_svd.real_compression_ratio
            else:
                patch_numel = x.shape[1] * self.patch_size[0] * self.patch_size[1]
                if rank == patch_numel:
                    z = x
                    self.real_compression_ratio = 1
                else:
                    z = self.patch_svd(x, rank=rank)
                    self.real_compression_ratio = self.patch_svd.real_compression_ratio
        return z

    def forward(self, x):
        z = self.transform(x)
        z = self.net(z)
        return z


class UVModel(nn.Module):
    def __init__(
        self,
        backbone=None,
        u_channels=1,
        v_channels=3,
        base_width=64,
        num_classes=1000,
        **kwargs,
    ):
        super().__init__()
        self.u_channels = u_channels
        self.v_channels = v_channels
        self.base_width = base_width
        self.num_classes = num_classes

        self.conv_u = nn.Conv3d(
            self.u_channels,
            self.base_width,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
        )
        self.conv_v = nn.Conv3d(
            self.v_channels,
            self.base_width,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
        )
        if backbone is None:
            self.backbone = ResNetBody(
                output_layer="layer4", spatial_dims=3, no_maxpool=True, **kwargs
            )
        else:
            self.backbone = backbone(**kwargs)

        self.fc = nn.LazyLinear(self.num_classes)

    def forward(self, x):
        u, v = x
        u = self.conv_u(u)
        u = self.backbone(u)

        v = self.conv_v(v)
        v = self.backbone(v)

        u = torch.mean(u, dim=(-2, -1))
        v = torch.mean(v, dim=(-2, -1))

        c = math.floor(2 ** (math.log2(u.shape[1]) // 2))
        u = rearrange(u, "b (c e) r -> b c e r", c=c)
        v = rearrange(v, "b (c e) r -> b c e r", c=c)

        y = torch.einsum("b c e r, b d e r -> b c d", u, v)
        y = torch.flatten(y, 1)
        y = self.fc(y)
        return y


class ResNetBody(nn.Module):
    def __init__(self, output_layer="layer4", **kwargs):
        super().__init__()
        resnet = resnet50(**kwargs)
        resnet.conv1 = nn.Identity()
        resnet.avgpool = nn.Identity()
        resnet.flatten = nn.Identity()
        resnet.fc = nn.Identity()
        # self.feature_extractor = create_feature_extractor(
        #     resnet, return_nodes={f"{output_layer}": "features"}
        # )
        self.body = resnet

    def forward(self, x):
        # features = self.feature_extractor(x)["features"]
        # return features
        out = self.body(x)
        return out
