from typing import List, Optional, Type, Union

import torch
import torch.nn as nn
from torch import Tensor

from torch.nn.modules.utils import _ntuple


CONV = {1: nn.Conv1d, 2: nn.Conv2d, 3: nn.Conv3d}
BATCH_NORM = {1: nn.BatchNorm1d, 2: nn.BatchNorm2d, 3: nn.BatchNorm3d}
MAXPOOL = {1: nn.MaxPool1d, 2: nn.MaxPool2d, 3: nn.MaxPool3d}
ADAPTIVE_AVGPOOL = {
    1: nn.AdaptiveAvgPool1d,
    2: nn.AdaptiveAvgPool2d,
    3: nn.AdaptiveAvgPool3d,
}


class BasicBlock(nn.Module):
    expansion: int = 1

    def __init__(
        self,
        in_planes: int,
        planes: int,
        spatial_dims: int = 2,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
    ) -> None:
        super().__init__()

        conv_type = CONV[spatial_dims]
        norm_type = BATCH_NORM[spatial_dims]

        self.conv1 = conv_type(
            in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn1 = norm_type(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv_type(
            planes, planes, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn2 = norm_type(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):

    expansion: int = 4

    def __init__(
        self,
        in_planes: int,
        planes: int,
        spatial_dims: int = 2,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
    ) -> None:
        super().__init__()
        conv_type = CONV[spatial_dims]
        norm_type = BATCH_NORM[spatial_dims]

        self.conv1 = conv_type(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = norm_type(planes)
        self.conv2 = conv_type(
            planes, planes, kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn2 = norm_type(planes)
        self.conv3 = conv_type(planes, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = norm_type(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class ResNet(nn.Module):
    def __init__(
        self,
        block: Type[Union[BasicBlock, Bottleneck]],
        layers: List[int],
        spatial_dims: int = 2,
        in_channels: int = 3,
        num_classes: int = 1000,
        conv1_kernel_size: int = 7,
        conv1_stride: int = 2,
        no_maxpool: bool = False,
        zero_init_residual: bool = False,
    ) -> None:
        super().__init__()
        self._conv_type = CONV[spatial_dims]
        self._norm_type = BATCH_NORM[spatial_dims]
        self._pool_type = MAXPOOL[spatial_dims]
        self._avgp_type = ADAPTIVE_AVGPOOL[spatial_dims]

        self.spatial_dims = spatial_dims
        self.in_channels = in_channels
        self.in_planes = 64
        self.no_maxpool = no_maxpool

        conv1_kernel_size = _ntuple(spatial_dims)(conv1_kernel_size)
        conv1_stride = _ntuple(spatial_dims)(conv1_stride)

        self.conv1 = self._conv_type(
            self.in_channels,
            self.in_planes,
            kernel_size=conv1_kernel_size,
            stride=conv1_stride,
            padding=tuple(k // 2 for k in conv1_kernel_size),
            bias=False,
        )
        self.bn1 = self._norm_type(self.in_planes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = self._pool_type(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = self._avgp_type(tuple(1 for i in range(spatial_dims)))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck) and m.bn3.weight is not None:
                    nn.init.constant_(m.bn3.weight, 0)  # type: ignore[arg-type]
                elif isinstance(m, BasicBlock) and m.bn2.weight is not None:
                    nn.init.constant_(m.bn2.weight, 0)  # type: ignore[arg-type]

    def _make_layer(
        self,
        block: Type[Union[BasicBlock, Bottleneck]],
        planes: int,
        blocks: int,
        stride: int = 1,
    ) -> nn.Sequential:
        if stride != 1 or self.in_planes != planes * block.expansion:
            downsample = nn.Sequential(
                self._conv_type(
                    self.in_planes,
                    planes * block.expansion,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                self._norm_type(planes * block.expansion),
            )
        else:
            downsample = None

        layers = [block(self.in_planes, planes, self.spatial_dims, stride, downsample)]
        self.in_planes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.in_planes, planes, self.spatial_dims))

        return nn.Sequential(*layers)
    
    def forward(self, x: Tensor) -> Tensor:
        x = self.conv1(x   )
        x = self.bn1(x)
        x = self.relu(x)
        if not self.no_maxpool:
            x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x


def resnet18(*args, **kwargs):
    return ResNet(BasicBlock, [2, 2, 2, 2], *args, **kwargs)


def resnet34(*args, **kwargs):
    return ResNet(BasicBlock, [3, 4, 6, 3], *args, **kwargs)


def resnet50(pretrained_weights_path=None, *args, **kwargs):
    model = ResNet(Bottleneck, [3, 4, 6, 3], *args, **kwargs)

    if pretrained_weights_path is not None:
        if torch.cuda.is_available():
            model.load_state_dict(torch.load(pretrained_weights_path))
        else:
            model.load_state_dict(torch.load(pretrained_weights_path, map_location=torch.device('cpu')))

    return model


def resnet101(*args, **kwargs):
    return ResNet(Bottleneck, [3, 4, 23, 3], *args, **kwargs)


def resnet152(*args, **kwargs):
    return ResNet(Bottleneck, [3, 8, 36, 3], *args, **kwargs)


# def test():
#     net = resnet50(
#         spatial_dims=3,
#         num_classes=10,
#         conv1_kernel_size=3,
#         conv1_stride=1,
#         no_maxpool=True,
#     )
#     y = net(torch.randn(1, 3, 32, 32, 32))
#     print(y.size())
