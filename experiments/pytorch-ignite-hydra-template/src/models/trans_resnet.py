from torch import nn
import torchvision

class transResnet(nn.Module):
    def __init__(
        self,
        net=None,
        num_classes=1000,
        **kwargs,
    ):
        super().__init__()

        if net is None:
            self.net = torchvision.models.resnet50(
                num_classes=num_classes, **kwargs
            )
        else:
            self.net = net(num_classes=num_classes, **kwargs)

    def forward(self, x):
        if isinstance(x, list or tuple):
            z = self.net(x[0])
            z.real_bpp = x[1]
        else:
            z = self.net(x)
        return z