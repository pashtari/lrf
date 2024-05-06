import numpy as np
import torch
from torchvision.transforms import v2
import matplotlib.pyplot as plt
from skimage.io import imread

import lrf

# Load the image
image = imread("./data/kodak/kodim23.png")

# Transform the input image
transforms = v2.Compose([v2.ToImage()])
image = torch.tensor(transforms(image))

# Visualize image
plt.imshow(image.permute(1, 2, 0))
plt.axis("off")
plt.title("Original Image")
plt.show()


# Store errors and compressed images for each method and CR/bpp
compression_ratios = {
    "JPEG": [],
    "WEBP": [],
    "SVD": [],
    "IMF - RGB": [],
    "IMF - YCbCr": [],
}

bpps = {
    "JPEG": [],
    "WEBP": [],
    "SVD": [],
    "IMF - RGB": [],
    "IMF - YCbCr": [],
}

reconstructed_images = {
    "JPEG": [],
    "WEBP": [],
    "SVD": [],
    "IMF - RGB": [],
    "IMF - YCbCr": [],
}

psnr_values = {
    "JPEG": [],
    "WEBP": [],
    "SVD": [],
    "IMF - RGB": [],
    "IMF - YCbCr": [],
}
ssim_values = {
    "JPEG": [],
    "WEBP": [],
    "SVD": [],
    "IMF - RGB": [],
    "IMF - YCbCr": [],
}

# Calculate reconstructed images and metric values for each method


# JPEG
for quality in range(0, 80, 1):
    enocoded = lrf.pil_encode(image, format="JPEG", quality=quality)
    reconstructed = lrf.pil_decode(enocoded)

    real_compression_ratio = lrf.get_compression_ratio(image, enocoded)
    real_bpp = lrf.get_bbp(image.shape[-2:], enocoded)

    compression_ratios["JPEG"].append(real_compression_ratio)
    bpps["JPEG"].append(real_bpp)
    reconstructed_images["JPEG"].append(reconstructed)
    psnr_values["JPEG"].append(lrf.psnr(image, reconstructed))
    ssim_values["JPEG"].append(lrf.ssim(image, reconstructed))


# # WebP
# for quality in range(0, 50, 1):
#     enocoded = lrf.pil_encode(image, format="WEBP", quality=quality, alpha_quality=0)
#     reconstructed = lrf.pil_decode(enocoded)

#     real_compression_ratio = lrf.get_compression_ratio(image, enocoded)
#     real_bpp = lrf.get_bbp(image.shape[-2:], enocoded)

#     compression_ratios["WEBP"].append(real_compression_ratio)
#     bpps["WEBP"].append(real_bpp)
#     reconstructed_images["WEBP"].append(reconstructed)
#     psnr_values["WEBP"].append(lrf.psnr(image, reconstructed))
#     ssim_values["WEBP"].append(lrf.ssim(image, reconstructed))

# SVD
for quality in np.linspace(0.0, 7, 30):
    enocoded = lrf.svd_encode(
        image,
        color_space="RGB",
        quality=quality,
        patch=True,
        patch_size=(8, 8),
        dtype=torch.int8,
    )
    reconstructed = lrf.svd_decode(enocoded)

    real_compression_ratio = lrf.get_compression_ratio(image, enocoded)
    real_bpp = lrf.get_bbp(image.shape[-2:], enocoded)

    compression_ratios["SVD"].append(real_compression_ratio)
    bpps["SVD"].append(real_bpp)
    reconstructed_images["SVD"].append(reconstructed)
    psnr_values["SVD"].append(lrf.psnr(image, reconstructed))
    ssim_values["SVD"].append(lrf.ssim(image, reconstructed))


# IMF - RGB
for quality in np.linspace(0.0, 30, 50):
    enocoded = lrf.imf_encode(
        image,
        color_space="RGB",
        quality=quality,
        patch=True,
        patch_size=(8, 8),
        bounds=(-16, 15),
        dtype=torch.int8,
        num_iters=10,
        verbose=False,
    )
    reconstructed = lrf.imf_decode(enocoded)

    real_compression_ratio = lrf.get_compression_ratio(image, enocoded)
    real_bpp = lrf.get_bbp(image.shape[-2:], enocoded)

    compression_ratios["IMF - RGB"].append(real_compression_ratio)
    bpps["IMF - RGB"].append(real_bpp)
    reconstructed_images["IMF - RGB"].append(reconstructed)
    psnr_values["IMF - RGB"].append(lrf.psnr(image, reconstructed))
    ssim_values["IMF - RGB"].append(lrf.ssim(image, reconstructed))


# IMF - YCbCr
for quality in np.linspace(0, 50, 50):
    enocoded = lrf.imf_encode(
        image,
        color_space="YCbCr",
        scale_factor=(0.5, 0.5),
        quality=(quality, quality / 2, quality / 2),
        patch=True,
        patch_size=(8, 8),
        bounds=(-16, 15),
        dtype=torch.int8,
        num_iters=10,
        verbose=False,
    )
    reconstructed = lrf.imf_decode(enocoded)

    real_compression_ratio = lrf.get_compression_ratio(image, enocoded)
    real_bpp = lrf.get_bbp(image.shape[-2:], enocoded)

    compression_ratios["IMF - YCbCr"].append(real_compression_ratio)
    bpps["IMF - YCbCr"].append(real_bpp)
    reconstructed_images["IMF - YCbCr"].append(reconstructed)
    psnr_values["IMF - YCbCr"].append(lrf.psnr(image, reconstructed))
    ssim_values["IMF - YCbCr"].append(lrf.ssim(image, reconstructed))


selected_methods = [
    "JPEG",
    # "WEBP",
    "SVD",
    "IMF - RGB",
    "IMF - YCbCr",
]
bpps = {k: bpps[k] for k in selected_methods}
reconstructed_images = {k: reconstructed_images[k] for k in selected_methods}
psnr_values = {k: psnr_values[k] for k in selected_methods}
ssim_values = {k: ssim_values[k] for k in selected_methods}

# Plotting the results: PSNR vs bpp
plt.figure()
for method, values in psnr_values.items():
    plt.plot(bpps[method], values, marker="o", markersize=4, label=method)

plt.xlabel("bpp")
plt.ylabel("PSNR (dB)")
plt.title("Comprison of Different Compression Methods")
# plt.xticks(np.arange(1, 13, 1))
plt.xlim(0.05, 0.5)
# plt.ylim(20, 30)
plt.legend()
plt.grid()
plt.savefig(
    "experiments/compression_methods_comparison_psnr-bpp.pdf",
    format="pdf",
    dpi=600,
)
plt.show()


# Plotting the results: SSIM vs bpp
plt.figure()
for method, values in ssim_values.items():
    plt.plot(bpps[method], values, marker="o", markersize=4, label=method)

plt.xlabel("bpp")
plt.ylabel("SSIM")
plt.title("Comprison of Different Compression Methods")
# plt.xticks(np.arange(1, 13, 1))
plt.xlim(0.05, 0.5)
# plt.ylim(0.5, 0.8)
plt.legend()
plt.grid()
plt.savefig(
    "experiments/compression_methods_comparison_ssim-bpp.pdf",
    format="pdf",
    dpi=600,
)
plt.show()


# Plotting the compressed images for each method and bpp
selected_bpps = [1, 0.75, 0.5, 0.4, 0.3, 0.2, 0.15, 0.1, 0.05]
fig, axs = plt.subplots(
    len(selected_bpps),
    len(selected_methods),
    figsize=(6 * len(selected_methods), 5 * len(selected_bpps)),
)

# Setting titles for columns
for ax, method in zip(axs[0], selected_methods):
    ax.set_title(method, fontsize=24)

for i, bbp in enumerate(selected_bpps):
    for j, method in enumerate(selected_methods):
        ii = np.argmin(np.abs(np.array(bpps[method]) - bbp))
        reconstructed_image = reconstructed_images[method][ii]
        axs[i, j].imshow(reconstructed_image.permute(1, 2, 0))
        real_bpp = bpps[method][ii]
        axs[i, j].set_ylabel(f"bpp = {real_bpp:.2f}", rotation=90, fontsize=18)
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