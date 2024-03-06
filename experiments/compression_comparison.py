import numpy as np
import torch
import matplotlib.pyplot as plt
from skimage import data, img_as_float
from skimage import exposure
from skimage.transform import resize

from src.models import compression as com


# Load the astronaut image
image = data.cat()

# Convert the image to float and resize
image = img_as_float(image)
image = resize(image, (224, 224), preserve_range=True)


# Visualize image
plt.imshow(image)
plt.axis("off")
plt.title("Astronaut Image")
plt.show()

# Convert to channel-first pytorch tensor
image = torch.tensor(image).permute(-1, 0, 1).unsqueeze(0)


# Store errors and compressed images for each method and compression ratio
compression_ratios = {
    "Interpolation": [],
    "Interpolation - Antialias": [],
    "Interpolation - Low": [],
    "DCT": [],
    "DCT - Low": [],
    "SVD": [],
    "Patch SVD": [],
    "LSD": [],
    "Patch LSD": [],
}

compressed_images = {
    "Interpolation": [],
    "Interpolation - Antialias": [],
    "Interpolation - Low": [],
    "DCT": [],
    "DCT - Low": [],
    "SVD": [],
    "Patch SVD": [],
    "LSD": [],
    "Patch LSD": [],
}

psnr_values = {
    "Interpolation": [],
    "Interpolation - Antialias": [],
    "DCT": [],
    "SVD": [],
    "Patch SVD": [],
    "LSD": [],
    "Patch LSD": [],
}
ssim_values = {
    "Interpolation": [],
    "Interpolation - Antialias": [],
    "DCT": [],
    "SVD": [],
    "Patch SVD": [],
    "LSD": [],
    "Patch LSD": [],
}

# Calculate reconstructed images and metric values for each method

# Interpolation
for new_size in range(32, 224, 16):
    interpolate = com.Interpolate(mode="bilinear", antialias=False)
    compressed_interpolate = interpolate(image, new_size=new_size)  # .clip(0, 1)
    compression_ratios["Interpolation"].append(interpolate.real_compression_ratio)
    compressed_images["Interpolation"].append(compressed_interpolate)
    psnr_values["Interpolation"].append(com.psnr(image, compressed_interpolate))
    ssim_values["Interpolation"].append(com.ssim(image, compressed_interpolate))

# Interpolation - Antialias
for new_size in range(32, 224, 16):
    interpolate = com.Interpolate(mode="bilinear", antialias=True)
    compressed_interpolate = interpolate(image, new_size=new_size)  # .clip(0, 1)
    compression_ratios["Interpolation - Antialias"].append(
        interpolate.real_compression_ratio
    )
    compressed_images["Interpolation - Antialias"].append(compressed_interpolate)
    psnr_values["Interpolation - Antialias"].append(
        com.psnr(image, compressed_interpolate)
    )
    ssim_values["Interpolation - Antialias"].append(
        com.ssim(image, compressed_interpolate)
    )

# Interpolation - low
for new_size in range(32, 224, 16):
    interpolate = com.Interpolate(mode="bilinear")
    compressed_interpolate_low = interpolate.compress(
        image, new_size=new_size
    )  # .clip(0, 1)
    compression_ratios["Interpolation - Low"].append(interpolate.real_compression_ratio)
    compressed_images["Interpolation - Low"].append(compressed_interpolate_low)

# DCT
for cutoff in range(32, 224, 16):
    dct = com.DCT()
    compressed_dct = dct(image, cutoff=cutoff)  # .clip(0, 1)
    compression_ratios["DCT"].append(dct.real_compression_ratio)
    compressed_images["DCT"].append(compressed_dct)
    psnr_values["DCT"].append(com.psnr(image, compressed_dct))
    ssim_values["DCT"].append(com.ssim(image, compressed_dct))

# DCT low
for cutoff in range(32, 224, 16):
    dct = com.DCT()
    compressed_dct_low = com.minmax_normalize(dct(image, cutoff=cutoff, pad=False))
    compression_ratios["DCT - Low"].append(dct.real_compression_ratio)
    compressed_images["DCT - Low"].append(compressed_dct_low)

# SVD
for rank in range(2, 224, 16):
    svd = com.SVD()
    compressed_svd = svd(image, rank=rank)  # .clip(0, 1)
    if svd.real_compression_ratio >= 1:
        compression_ratios["SVD"].append(svd.real_compression_ratio)
        compressed_images["SVD"].append(compressed_svd)
        psnr_values["SVD"].append(com.psnr(image, compressed_svd))
        ssim_values["SVD"].append(com.ssim(image, compressed_svd))

# Patch SVD
for rank in range(3, 193, 13):
    patch_svd = com.PatchSVD(patch_size=(8, 8))
    compressed_patch_svd = patch_svd(image, rank=rank)  # .clip(0, 1)
    if patch_svd.real_compression_ratio >= 1:
        compression_ratios["Patch SVD"].append(patch_svd.real_compression_ratio)
        compressed_images["Patch SVD"].append(compressed_patch_svd)
        psnr_values["Patch SVD"].append(com.psnr(image, compressed_patch_svd))
        ssim_values["Patch SVD"].append(com.ssim(image, compressed_patch_svd))

# LSD
alpha = 0.005
for rank in range(2, 224, 16):
    lsd = com.LSD(num_iters=10)
    compressed_lsd = lsd(image, rank=rank, alpha=alpha)  # .clip(0, 1)
    if lsd.real_compression_ratio >= 1:
        compression_ratios["LSD"].append(lsd.real_compression_ratio)
        compressed_images["LSD"].append(compressed_lsd)
        psnr_values["LSD"].append(com.psnr(image, compressed_lsd))
        ssim_values["LSD"].append(com.ssim(image, compressed_lsd))

# Patch LSD
alpha = 0.005
for rank in range(3, 192, 13):
    patch_lsd = com.PatchLSD(num_iters=10, verbose=True)
    compressed_patch_lsd = patch_lsd(image, rank=rank, alpha=alpha)  # .clip(0, 1)
    if patch_lsd.real_compression_ratio >= 1:
        compression_ratios["Patch LSD"].append(patch_lsd.real_compression_ratio)
        compressed_images["Patch LSD"].append(compressed_patch_lsd)
        psnr_values["Patch LSD"].append(com.psnr(image, compressed_patch_lsd))
        ssim_values["Patch LSD"].append(com.ssim(image, compressed_patch_lsd))


# Plotting the results: PSNR
plt.figure()
for method, values in psnr_values.items():
    plt.plot(compression_ratios[method], values, marker="o", label=method)

plt.xlabel("Compression Ratio")
plt.ylabel("PSNR")
plt.title("Comprison of Different Compression Methods")
plt.xticks(np.arange(1, max(compression_ratios["Patch LSD"]) + 1, 3))
plt.legend()
plt.grid()
plt.savefig(
    "experiments/compression_methods_comparison_psnr.pdf",
    format="pdf",
    dpi=600,
)
plt.show()

# Plotting the results: SSIM
plt.figure()
for method, values in ssim_values.items():
    plt.plot(compression_ratios[method], values, marker="o", label=method)

plt.xlabel("Compression Ratio")
plt.ylabel("SSIM")
plt.title("Comprison of Different Compression Methods")
plt.xticks(np.arange(1, max(compression_ratios["Patch LSD"]) + 1, 3))
plt.legend()
plt.grid()
plt.savefig(
    "experiments/compression_methods_comparison_ssim.pdf",
    format="pdf",
    dpi=600,
)
plt.show()


# Plotting the compressed images for each method and compression ratio
selected_ratios = [2, 5, 10, 15, 20, 25]
selected_methods = ["Interpolation", "DCT", "Patch SVD", "Patch LSD"]
fig, axs = plt.subplots(
    len(selected_ratios),
    len(selected_methods),
    figsize=(5 * len(selected_methods), 5 * len(selected_ratios)),
)

# Setting titles for columns
for ax, method in zip(axs[0], selected_methods):
    ax.set_title(method, fontsize=24)

for i, ratio in enumerate(selected_ratios):
    for j, method in enumerate(selected_methods):
        ii = np.argmin(np.abs(np.array(compression_ratios[method]) - ratio))
        compressed_image = compressed_images[method][ii]
        axs[i, j].imshow(compressed_image.squeeze(0).permute(1, 2, 0))
        real_ratio = compression_ratios[method][ii]
        axs[i, j].set_ylabel(f"Ratio = {real_ratio:.2f}", rotation=90, fontsize=18)
        axs[i, j].tick_params(
            axis="both",
            which="both",
            bottom=False,
            left=False,
            labelbottom=False,
            labelleft=False,
        )


plt.tight_layout()
plt.savefig(
    "experiments/compression_methods_qualitative_comparison.pdf",
    format="pdf",
    dpi=600,
)
plt.show()
