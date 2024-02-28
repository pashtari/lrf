import torch
from deepspeed.profiling.flops_profiler import get_model_profile
import matplotlib.pyplot as plt

from src.models import InterpolateModel, PatchSVDModel

# Dummy input tensor
x = torch.rand(1, 3, 224, 224)

# Interpolate
interpolate_flops = []
interpolate_ratios = []
for new_size in range(140, 141):
    interpolate_model = InterpolateModel(new_size=new_size, rescale=False)
    z = interpolate_model.transform(x)
    flops, macs, params = get_model_profile(
        interpolate_model.net, args=(z,), as_string=False
    )
    interpolate_flops.append(flops * 1e-9)
    interpolate_ratios.append(interpolate_model.interpolate.real_compression_ratio)

# PatchSVD
patchsvd_flops = []
patchsvd_ratios = []
for rank in range(1, 2):
    patchsvd_model = PatchSVDModel(rank=rank, patch_size=8, domain="compressed")
    z = u, v = patchsvd_model.transform(x)
    flops, macs, params = get_model_profile(
        patchsvd_model.net, args=(z,), as_string=False
    )
    patchsvd_flops.append(flops * 1e-9)
    patchsvd_ratios.append(patchsvd_model.patch_svd.real_compression_ratio)


# Plotting
plt.rcParams.update({"font.size": 14})
plt.figure(figsize=(10, 6))
plt.plot(interpolate_ratios, interpolate_flops, label="Interpolate", marker="o")
plt.plot(patchsvd_ratios, patchsvd_flops, label="PatchSVD", marker="s")


# plt.xticks()
plt.xlabel("Compression Ratio")
plt.ylabel("FLOPs (G)")
plt.title("Comparison of Methods Complexity Across Different Compression Ratios")
plt.legend()
plt.grid(False)
plt.savefig("experiments/comparison_of_methods_complexity.pdf", format="pdf", dpi=600)
plt.show()
