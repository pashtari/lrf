from typing import Iterable, Sequence
import math
import random

import torch
from torch import nn
from torch.nn.modules.utils import _pair
import torchvision
from einops import rearrange

from .compression import PatchSVD, zscore_normalize
from .resnet import resnet, Bottleneck3d
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
                self.net = net(num_classes=self.num_classes, **kwargs)

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
                u, v = self.patch_svd.compress(x, rank=rank)
                u, v = self.patch_svd.depatchify_uv(x, u, v)
                self.real_compression_ratio = self.patch_svd.real_compression_ratio
                z = zscore_normalize(u), zscore_normalize(v)
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

        self.conv1_u = nn.Conv3d(
            in_channels=u_channels,
            out_channels=self.base_width,
            kernel_size=(3, 3, 3),
            stride=(2, 2, 2),
            padding=(1, 1, 1),
            bias=False,
        )
        self.bn1_u = nn.BatchNorm3d(self.base_width)
        self.conv2_u = nn.Conv3d(
            in_channels=self.base_width,
            out_channels=self.base_width,
            kernel_size=(3, 1, 1),
            stride=(2, 1, 1),
            padding=(1, 0, 0),
            bias=False,
        )
        self.bn2_u = nn.BatchNorm3d(self.base_width)

        self.conv1_v = nn.Conv3d(
            in_channels=v_channels,
            out_channels=self.base_width,
            kernel_size=(3, 3, 3),
            stride=(2, 1, 1),
            padding=(1, 1, 1),
            bias=False,
        )
        self.bn1_v = nn.BatchNorm3d(self.base_width)
        self.conv2_v = nn.Conv3d(
            in_channels=self.base_width,
            out_channels=self.base_width,
            kernel_size=(3, 1, 1),
            stride=(2, 1, 1),
            padding=(1, 0, 0),
            bias=False,
        )
        self.bn2_v = nn.BatchNorm3d(self.base_width)

        if backbone is None:
            self.backbone = ResNetBody(
                net=resnet,
                block=Bottleneck3d,
                layers=[3, 4, 6, 3],
                spatial_dims=3,
                **kwargs,
            )
        else:
            self.backbone = backbone(**kwargs)

        self.flatten = nn.Flatten(start_dim=1)
        self.bn_out = nn.BatchNorm1d(num_features=1024)
        self.fc = nn.Linear(in_features=1024, out_features=self.num_classes)

    def forward(self, x: torch.Tensor):
        # u, v = x  # u: B × 1 × R × H × W,    v: B × 3 × R × P × P
        # u, v = u.flatten(1, 2), v.flatten(1, 2)

        u, v = x  # u: B × R × 1 × H × W,    v: B × R × 3 × P × P

        # print(f"u.mean(), u.std(): ({u.mean()}, {u.std()})")
        u = self.conv1_u(u)
        u = self.bn1_u(u)
        # print(f"bn1_u: u.mean(), u.std(): ({u.mean()}, {u.std()})")
        u = self.conv2_u(u)
        u = self.bn2_u(u)
        # print(f"bn2_u: u.mean(), u.std(): ({u.mean()}, {u.std()})")
        u = self.backbone(u)
        # print(f"backbone_u: u.mean(), u.std(): ({u.mean()}, {u.std()})")

        # print(f"v.mean(), v.std(): ({v.mean()}, {v.std()})")
        v = self.conv1_v(v)
        v = self.bn1_v(v)
        # print(f"bn1_v: v.mean(), v.std(): ({v.mean()}, {v.std()})")
        v = self.conv2_v(v)
        v = self.bn2_v(v)
        # print(f"bn2_v: v.mean(), v.std(): ({v.mean()}, {v.std()})")
        v = self.backbone(v)
        # print(f"backbone_v: v.mean(), v.std(): ({v.mean()}, {v.std()})")

        u = torch.mean(u, dim=(-2, -1))
        # print(f"torch.mean_u: u.mean(), u.std(): ({u.mean()}, {u.std()})")
        v = torch.mean(v, dim=(-2, -1))
        # print(f"torch.mean_v: v.mean(), v.std(): ({v.mean()}, {v.std()})")

        # c = math.floor(2 ** (math.log2(u.shape[1]) // 2))
        # u = rearrange(u, "b (g c) -> b g c", c=c)
        # v = rearrange(v, "b (g c) -> b g c", c=c)

        # y = torch.einsum("b f c, b g c -> b f g", u, v)

        c = math.floor(2 ** (math.log2(u.shape[1]) // 2))
        u = rearrange(u, "b (c e) r -> b c e r", c=c)
        # print(f"rearrange_u: u.mean(), u.std(): ({u.mean()}, {u.std()})")
        v = rearrange(v, "b (c e) r -> b c e r", c=c)
        # print(f"rearrange_v: v.mean(), v.std(): ({v.mean()}, {v.std()})")

        y = torch.einsum("b c e r, b d e r -> b c d", u, v)
        # print(f"einsum_y: y.mean(), y.std(): ({y.mean()}, {y.std()})")

        y = self.flatten(y)
        # print(f"flatten_y: y.mean(), y.std(): ({y.mean()}, {y.std()})")

        y = self.bn_out(y)
        # print(f"bn_out_y: y.mean(), y.std(): ({y.mean()}, {y.std()})")

        y = self.fc(y)
        # print(f"fc_y: y.mean(), y.std(): ({y.mean()}, {y.std()})")
        return y


class ResNetBody(nn.Module):
    def __init__(self, net, **kwargs):
        super().__init__()

        model = net(**kwargs)
        self.layer1 = model.layer1
        self.layer2 = model.layer2
        self.layer3 = model.layer3
        self.layer4 = model.layer4

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        return out
