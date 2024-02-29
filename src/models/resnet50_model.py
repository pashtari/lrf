from torch import nn
from torchvision.models import resnet50

class Resnet50Model(nn.Module):
    def __init__(
        self,
        **kwargs
    ):
        super(Resnet50Model, self).__init__()
        self.net = resnet50(**kwargs)

    def forward(self, x):
        y = self.net(x)
        return y
