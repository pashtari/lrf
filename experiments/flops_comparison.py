import torch
from torch import nn
from fvcore.nn import FlopCountAnalysis
import matplotlib.pyplot as plt

from src.models import ResNet, SVDResNet

# Define a range of compression ratios
compression_ratios = range(1, 21)  # From 1 to 20

# Initialize lists to store FLOPs for each module
resnet_flops = []
svdresnet4_flops = []
svdresnet8_flops = []
svdresnet16_flops = []
conv_flops = []

# Dummy input tensor
x = torch.rand(1, 3, 224, 224)

# Loop over each compression ratio
for ratio in compression_ratios:
    # ResNet
    resnet = ResNet(compression_ratio=ratio, rescale=False)
    z = resnet.compress(x)
    macs = FlopCountAnalysis(resnet.model, z).total()
    flops = 2 * macs * 1e-9
    resnet_flops.append(flops)

    # SVDResNet
    svdresnet = SVDResNet(compression_ratio=ratio, patch_size=4)
    u, v = svdresnet.compress(x)
    macs = FlopCountAnalysis(svdresnet.model, (u, v)).total()
    flops = 2 * macs * 1e-9
    svdresnet4_flops.append(flops)

    # SVDResNet
    svdresnet = SVDResNet(compression_ratio=ratio, patch_size=8)
    u, v = svdresnet.compress(x)
    macs = FlopCountAnalysis(svdresnet.model, (u, v)).total()
    flops = 2 * macs * 1e-9
    svdresnet8_flops.append(flops)

    # SVDResNet
    svdresnet = SVDResNet(compression_ratio=ratio, patch_size=16)
    u, v = svdresnet.compress(x)
    macs = FlopCountAnalysis(svdresnet.model, (u, v)).total()
    flops = 2 * macs * 1e-9
    svdresnet16_flops.append(flops)


# Plotting
plt.figure(figsize=(10, 6))
plt.plot(compression_ratios, resnet_flops, label="ResNet", marker="o")
plt.plot(compression_ratios, svdresnet4_flops, label="SVDResNet (p=4)", marker="s")
plt.plot(compression_ratios, svdresnet8_flops, label="SVDResNet (p=8)", marker="s")
plt.plot(compression_ratios, svdresnet16_flops, label="SVDResNet (p=16)", marker="s")
plt.xticks(compression_ratios)
plt.xlabel("Compression Ratio")
plt.ylabel("FLOPs (G)")
plt.title("Comparison of Methods Complexity for Different Compression Ratios")
plt.legend()
plt.grid(False)
plt.show()
