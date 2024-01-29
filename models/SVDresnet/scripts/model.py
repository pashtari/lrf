from torch import nn
from torchvision.models import resnet50
from torchvision.models.feature_extraction import create_feature_extractor
from einops import rearrange 
from utilities import get_svd_patches_u, get_svd_patches_v # ?? where/how to include these utilities?

class SVDResNet(nn.Module):
    def __init__(self, compression_ratio):
        super(SVDResNet, self).__init__()

        num_classes=10
        self.compression_ratio = compression_ratio # different cases for random, [min, max], fixed, etc. to be added
        model = resnet50()
        model.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False) # making it suitable for one channel

        self.resnet_convs = create_feature_extractor(model, {"flatten": "flatten"}) # all the convolution layers of resnet50
        self.linear_mapping_1 = nn.Linear(2048, 64) # ?? hyperparams in train.yaml?
        self.linear_mapping_2 = nn.Linear(192, 16)
        self.resnet_fc_layer = nn.Sequential(
            nn.BatchNorm1d(1024),
            nn.Linear(1024, num_classes)
        )


    def forward(self, x):
        batch_size = x.shape[0]
        svd_patches_u = get_svd_patches_u(x, self.compression_ratio)
        svd_patches_v = get_svd_patches_v(x, self.compression_ratio)

        U = self.resnet_convs(svd_patches_u)["flatten"]
        U = self.linear_mapping_1(U)
        V = self.linear_mapping_2(svd_patches_v)

        U = rearrange(U, "(b r) d1 -> b d1 r", b=batch_size)
        V = rearrange(V, "(b r) d1 -> b r d1", b=batch_size)

        X = U @ V
        X = X.view(batch_size, -1)
        X = self.resnet_fc_layer(X)

        return X
    
