import math
import random
from contextlib import contextmanager

import torch
from torch import nn
from torch.nn.modules.utils import _pair
from torchvision.models import resnet34, resnet50
from torchvision.models.feature_extraction import create_feature_extractor
from einops import rearrange


@contextmanager
def null_context():
    yield


def get_rank(shape, compression_ratio):
    m, n = shape
    rank = math.floor(m * n // (compression_ratio * (m + n)))
    return rank


def get_patch_svd_components(x, patch_size=(8, 8), compression_ratio=1):
    # x: B × C × H × W
    p1, p2 = patch_size
    # Rearrange image to form rows of flattened patches
    patches = rearrange(x, "b c (h p1) (w p2) -> b (h w) (c p1 p2)", p1=p1, p2=p2)

    r = get_rank(patches.shape[-2:], compression_ratio)

    u, s, v = torch.linalg.svd(patches, full_matrices=False)
    u = u[:, :, :r] @ torch.diag_embed(torch.sqrt(s[:, :r]))
    v = torch.diag_embed(torch.sqrt(s[:, :r])) @ v[:, :r, :]
    # patches ≈ u @ v

    u = rearrange(
        u,
        "b (h w) r -> b r 1 h w",
        h=x.shape[-2] // p1,
    )
    # v: b r (c p1 p2)
    return u, v


class SVDResNet(nn.Module):
    def __init__(
        self,
        channels=3,
        patch_size=8,
        compression_ratio=1,
        u_output_dim=96,
        v_output_dim=24,
        num_classes=1000,
        no_grad=True,
    ):
        super(SVDResNet, self).__init__()

        self.channels = channels
        self.num_classes = num_classes
        p1, p2 = self.patch_size = _pair(patch_size)
        self.v_input_dim = channels * p1 * p2
        self.u_output_dim = u_output_dim
        self.v_output_dim = v_output_dim
        self.no_grad = no_grad
        self.compression_ratio = _pair(compression_ratio)
        self.model = UVModel(
            u_input_dim=1,
            v_input_dim=self.v_input_dim,
            u_output_dim=u_output_dim,
            v_output_dim=v_output_dim,
            num_classes=num_classes,
        )

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

        u, v = get_patch_svd_components(x, self.patch_size, cr)

        return u, v

    def forward(self, x):
        with self.context():
            u, v = self.compress(x)

        y = self.model(u, v)
        return y


class UVModel(nn.Module):
    def __init__(
        self,
        u_input_dim,
        v_input_dim,
        u_output_dim,
        v_output_dim,
        num_classes,
    ):
        super(UVModel, self).__init__()
        self.num_classes = num_classes
        self.u_input_dim = u_input_dim
        self.v_input_dim = v_input_dim
        self.u_output_dim = u_output_dim
        self.v_output_dim = v_output_dim

        self.resnet = resnet50()
        self.resnet.conv1 = nn.Conv2d(
            self.u_input_dim, 64, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.net = create_feature_extractor(self.resnet, {"flatten": "features"})
        self.linear_u = nn.Linear(2048, self.u_output_dim)
        self.linear_v = nn.Linear(self.v_input_dim, self.v_output_dim)

        self.fc = nn.Linear(self.u_output_dim * self.v_output_dim, self.num_classes)

    def forward(self, u, v):
        b = u.shape[0]
        u = rearrange(u, "b r c h w -> (b r) c h w")
        v = rearrange(v, "b r d -> (b r) d")

        u = self.net(u)["features"]
        u = self.linear_u(u)
        v = self.linear_v(v)

        u = rearrange(u, "(b r) d -> b d r", b=b)
        v = rearrange(v, "(b r) d -> b r d", b=b)

        y = u @ v

        y = torch.flatten(y, 1)

        y = self.fc(y)
        return y
