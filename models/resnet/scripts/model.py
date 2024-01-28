from torch import nn
from torchvision.models import resnet50

class SVDResNet(nn.Module):
    def __init__(self):
        super(SVDResNet, self).__init__()
        
        self.model = resnet50()

    def forward(self, x):
        return self.model(x)