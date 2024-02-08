import torch
from deepspeed.profiling.flops_profiler import get_model_profile
import matplotlib.pyplot as plt

from src.models import ResNet, SVDResNet

# Define a range of compression ratios
compression_ratios = range(1, 21)  # From 1 to 20

# Specify a list of patch sizes for SVDResNet
patch_sizes = [4, 8, 16, 32]

# Initialize lists to store FLOPs for each module
resnet_flops = []
svdresnet_flops = {
    p: [] for p in patch_sizes
}  # Dictionary to hold FLOPs for different patch sizes


# Dummy input tensor
x = torch.rand(1, 3, 224, 224)

# Loop over each compression ratio
for ratio in compression_ratios:
    # ResNet
    resnet = ResNet(compression_ratio=ratio, rescale=False)
    z = resnet.compress(x)
    flops, macs, params = get_model_profile(resnet.model, args=(z,), as_string=False)
    resnet_flops.append(flops * 1e-9)

    # Loop over specified patch sizes for SVDResNet
    for patch_size in patch_sizes:
        svdresnet = SVDResNet(compression_ratio=ratio, patch_size=patch_size)
        u, v = svdresnet.compress(x)
        flops, macs, params = get_model_profile(
            svdresnet.model, args=(u, v), as_string=False
        )
        svdresnet_flops[patch_size].append(flops * 1e-9)

# Plotting
plt.rcParams.update({"font.size": 14})
plt.figure(figsize=(10, 6))
plt.plot(compression_ratios, resnet_flops, label="ResNet", marker="o")

# Plot for each patch size in SVDResNet
for patch_size, flops in svdresnet_flops.items():
    plt.plot(compression_ratios, flops, label=f"SVDResNet (p={patch_size})", marker="s")

plt.xticks(compression_ratios)
plt.xlabel("Compression Ratio")
plt.ylabel("FLOPs (G)")
plt.title("Comparison of Methods Complexity Across Different Compression Ratios")
plt.legend()
plt.grid(False)
plt.savefig("experiments/comparison_of_methods_complexity.pdf", format="pdf", dpi=600)
plt.show()
