import math
import random

from torch import nn
from torch.nn.modules.utils import _pair
from torch.nn.functional import interpolate
from torchvision.models import resnet50

class IMGResNet_rescale(nn.Module):
    def __init__(
        self,
        num_classes=10,
        compression_ratio=1,
        image_size=(224,224),
    ):
        super(IMGResNet_rescale, self).__init__()

        self.num_classes = num_classes
        self.compression_ratio = _pair(compression_ratio)
        self.image_size = _pair(image_size)
        self.model = resnet50()

    def forward(self, x):
        cr_sqrt = math.sqrt(random.uniform(*self.compression_ratio))

        x = interpolate(x, scale_factor=1/cr_sqrt)
        x = interpolate(x, size=self.image_size)
        x = self.model(x)

        return x
