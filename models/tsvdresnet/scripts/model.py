import math
import random

import torch
from torch import nn
from torch.nn.modules.utils import _pair
from torchvision.models import resnet50
from torchvision.models.feature_extraction import create_feature_extractor
from einops import rearrange


def get_patch_svd_components(x, patch_size=(8, 8), compression_ratio=1):
    # x: B × C × H × W
    p1, p2 = patch_size
    # Rearrange image to form columns of flattened patches
    patches = rearrange(x, "b c (h p1) (w p2) -> b (h w) (c p1 p2)", p1=p1, p2=p2)

    M, N = patches.shape[-2:]
    R = math.floor(M * N // (compression_ratio * (M + N)))

    u, s, vt = torch.linalg.svd(patches, full_matrices=False)
    u = u[:, :, :R] @ torch.diag_embed(torch.sqrt(s[:, :R]))
    vt = torch.diag_embed(torch.sqrt(s[:, :R])) @ vt[:, :R, :]

    u_reshaped = rearrange(
        u,
        "b (h w) r -> (b r) 1 h w",
        h=x.shape[-2] // p1,
    )
    vt_reshaped = rearrange(
        vt,
        "b r (c p1 p2) -> (b r) (c p1 p2)",
        p1=p1,
        p2=p2,
    )
    return u_reshaped, vt_reshaped


class SVDResNet(nn.Module):
    def __init__(
        self,
        num_classes=10,
        patch_size=8,
        compression_ratio=1,
        u_output_dim=96,
        vt_output_dim=24,
    ):
        super(SVDResNet, self).__init__()

        self.num_classes = num_classes
        p1, p2 = self.patch_size = _pair(patch_size)
        self.compression_ratio = _pair(compression_ratio)
        self.u_output_dim = u_output_dim
        self.vt_output_dim = vt_output_dim

        self.model = resnet50()
        self.model.conv1 = nn.Conv2d(
            1, 64, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.resnet = create_feature_extractor(self.model, {"avgpool": "features"})
        self.linear_u = nn.Linear(2048, self.u_output_dim)
        self.linear_vt = nn.Linear(3 * p1 * p2, self.vt_output_dim)

        self.fc = nn.Linear(self.u_output_dim * self.vt_output_dim, self.num_classes)

    def forward(self, x):
        batch_size = x.shape[0]

        cr = random.uniform(*self.compression_ratio)
        u, vt = get_patch_svd_components(x, self.patch_size, cr)

        u = self.resnet(u)["features"]
        u = self.linear_u(u)
        u = rearrange(u, "(b r) d -> b d r", b=batch_size)

        vt = self.linear_vt(vt)
        vt = rearrange(vt, "(b r) d -> b r d", b=batch_size)

        y = u @ vt
        y = torch.flatten(y, 1)
        y = self.fc(y)
        return y
