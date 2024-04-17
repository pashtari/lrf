import numpy as np
import torch
from torchvision.transforms import v2
import matplotlib.pyplot as plt
from skimage import data
from skimage.io import imread

from src.models import compression as com


# Load the astronaut image
image = data.astronaut()
image = imread(
    "/Users/pooya/Library/CloudStorage/OneDrive-KULeuven/research/projects/lsvd/data/kodak/kodim15.png"
)


# transforms = v2.Compose([v2.ToImage(), v2.Resize(size=(224, 224), interpolation=2)])
transforms = v2.Compose([v2.ToImage()])


# Transform the input image
image = torch.tensor(transforms(image))


# Visualize image
plt.imshow(image.permute(1, 2, 0))
plt.axis("off")
plt.title("Original Image")
plt.show()


# Store errors and compressed images for each method and CR/bpp
compression_ratios = {
    "JPEG": [],
    "JPEG2000": [],
    "WEBP": [],
    "SVD": [],
    "Patch SVD": [],
    "Patch IMF - RGB": [],
    "Patch IMF - YCbCr": [],
    "HOSVD": [],
    "Patch HOSVD": [],
}

bpps = {
    "JPEG": [],
    "JPEG2000": [],
    "WEBP": [],
    "SVD": [],
    "Patch SVD": [],
    "Patch IMF - RGB": [],
    "Patch IMF - YCbCr": [],
    "HOSVD": [],
    "Patch HOSVD": [],
}

reconstructed_images = {
    "JPEG": [],
    "JPEG2000": [],
    "WEBP": [],
    "SVD": [],
    "Patch SVD": [],
    "Patch IMF - RGB": [],
    "Patch IMF - YCbCr": [],
    "HOSVD": [],
    "Patch HOSVD": [],
}

psnr_values = {
    "JPEG": [],
    "JPEG2000": [],
    "WEBP": [],
    "SVD": [],
    "Patch SVD": [],
    "Patch IMF - RGB": [],
    "Patch IMF - YCbCr": [],
    "HOSVD": [],
    "Patch HOSVD": [],
}
ssim_values = {
    "JPEG": [],
    "JPEG2000": [],
    "WEBP": [],
    "SVD": [],
    "Patch SVD": [],
    "Patch IMF - RGB": [],
    "Patch IMF - YCbCr": [],
    "HOSVD": [],
    "Patch HOSVD": [],
}

# Calculate reconstructed images and metric values for each method


# JPEG
for quality in range(0, 30, 1):
    enocoded = com.pil_encode(image, format="JPEG", quality=quality)
    reconstructed = com.pil_decode(enocoded)

    real_compression_ratio = com.get_compression_ratio(image, enocoded)
    real_bpp = com.get_bbp(image.shape[-2:], enocoded)

    compression_ratios["JPEG"].append(real_compression_ratio)
    bpps["JPEG"].append(real_bpp)
    reconstructed_images["JPEG"].append(reconstructed)
    psnr_values["JPEG"].append(com.psnr(image, reconstructed))
    ssim_values["JPEG"].append(com.ssim(image, reconstructed))


# # JPEG2000
# for quality in range(1, 95, 2):
#     enocoded = com.pil_encode(image, format="JPEG2000", quality=quality)
#     reconstructed = com.pil_decode(enocoded)

#     real_compression_ratio = com.get_compression_ratio(image, enocoded)
#     real_bpp = com.get_bbp(image.shape[-2:], enocoded)

#     compression_ratios["JPEG2000"].append(real_compression_ratio)
#     bpps["JPEG2000"].append(real_bpp)
#     reconstructed_images["JPEG2000"].append(reconstructed)
#     psnr_values["JPEG2000"].append(com.psnr(image, reconstructed))
#     ssim_values["JPEG2000"].append(com.ssim(image, reconstructed))

# # WebP
# for quality in range(0, 50, 1):
#     enocoded = com.pil_encode(image, format="WEBP", quality=quality, alpha_quality=0)
#     reconstructed = com.pil_decode(enocoded)

#     real_compression_ratio = com.get_compression_ratio(image, enocoded)
#     real_bpp = com.get_bbp(image.shape[-2:], enocoded)

#     compression_ratios["WEBP"].append(real_compression_ratio)
#     bpps["WEBP"].append(real_bpp)
#     reconstructed_images["WEBP"].append(reconstructed)
#     psnr_values["WEBP"].append(com.psnr(image, reconstructed))
#     ssim_values["WEBP"].append(com.ssim(image, reconstructed))

# # SVD
# for quality in np.linspace(0.0, 0.08, 20):
#     enocoded = com.svd_encode(image, quality=quality, dtype=torch.int8)
#     reconstructed = com.svd_decode(enocoded)

#     real_compression_ratio = com.get_compression_ratio(image, enocoded)
#     real_bpp = com.get_bbp(image.shape[-2:], enocoded)

#     compression_ratios["SVD"].append(real_compression_ratio)
#     bpps["SVD"].append(real_bpp)
#     reconstructed_images["SVD"].append(reconstructed)
#     psnr_values["SVD"].append(com.psnr(image, reconstructed))
#     ssim_values["SVD"].append(com.ssim(image, reconstructed))


# Patch SVD
for quality in np.linspace(0.0, 0.05, 20):
    enocoded = com.patch_svd_encode(
        image, quality=quality, patch_size=(8, 8), dtype=torch.int8
    )
    reconstructed = com.patch_svd_decode(enocoded)

    real_compression_ratio = com.get_compression_ratio(image, enocoded)
    real_bpp = com.get_bbp(image.shape[-2:], enocoded)

    compression_ratios["Patch SVD"].append(real_compression_ratio)
    bpps["Patch SVD"].append(real_bpp)
    reconstructed_images["Patch SVD"].append(reconstructed)
    psnr_values["Patch SVD"].append(com.psnr(image, reconstructed))
    ssim_values["Patch SVD"].append(com.ssim(image, reconstructed))


# Patch IMF - RGB
for quality in np.linspace(0.0, 0.1, 50):
    enocoded = com.patch_imf_encode(
        image,
        color_space="RGB",
        quality=quality,
        patch_size=(8, 8),
        bounds=(-16, 15),
        dtype=torch.int8,
        num_iters=10,
        verbose=False,
    )
    reconstructed = com.patch_imf_decode(enocoded, color_space="RGB")

    real_compression_ratio = com.get_compression_ratio(image, enocoded)
    real_bpp = com.get_bbp(image.shape[-2:], enocoded)

    compression_ratios["Patch IMF - RGB"].append(real_compression_ratio)
    bpps["Patch IMF - RGB"].append(real_bpp)
    reconstructed_images["Patch IMF - RGB"].append(reconstructed)
    psnr_values["Patch IMF - RGB"].append(com.psnr(image, reconstructed))
    ssim_values["Patch IMF - RGB"].append(com.ssim(image, reconstructed))


# Patch IMF - YCbCr
for quality in np.linspace(0.0, 0.25, 50):
    enocoded = com.patch_imf_encode(
        image,
        color_space="YCbCr",
        scale_factor=(0.5, 0.5),
        quality=(quality, quality / 2, quality / 2),
        patch_size=(8, 8),
        bounds=(-16, 15),
        dtype=torch.int8,
        num_iters=10,
        verbose=False,
    )
    reconstructed = com.patch_imf_decode(enocoded, color_space="YCbCr")

    real_compression_ratio = com.get_compression_ratio(image, enocoded)
    real_bpp = com.get_bbp(image.shape[-2:], enocoded)

    compression_ratios["Patch IMF - YCbCr"].append(real_compression_ratio)
    bpps["Patch IMF - YCbCr"].append(real_bpp)
    reconstructed_images["Patch IMF - YCbCr"].append(reconstructed)
    psnr_values["Patch IMF - YCbCr"].append(com.psnr(image, reconstructed))
    ssim_values["Patch IMF - YCbCr"].append(com.ssim(image, reconstructed))


# # HOSVD
# for rank in range(1, 63, 4):
#     enocoded = com.hosvd_encode(image, rank=(3, rank, rank), dtype=torch.int16)
#     reconstructed = com.hosvd_decode(enocoded)

#     real_compression_ratio = com.get_compression_ratio(image, enocoded)
#     real_bpp = com.get_bbp(image.shape[-2:], enocoded)

#     compression_ratios["HOSVD"].append(real_compression_ratio)
#     bpps["HOSVD"].append(real_bpp)
#     reconstructed_images["HOSVD"].append(reconstructed)
#     psnr_values["HOSVD"].append(com.psnr(image, reconstructed))
#     ssim_values["HOSVD"].append(com.ssim(image, reconstructed))


# # Patch HOSVD
# for b in np.linspace(0.15, 6.25, 15):
#     print(f"HOSVD: bpp: {b:.2f}")
#     enocoded = com.patch_hosvd_encode(image, bpp=b, patch_size=(8, 8), dtype=torch.int16)
#     reconstructed = com.patch_hosvd_decode(enocoded)

#     real_compression_ratio = com.get_compression_ratio(image, enocoded)
#     real_bpp = com.get_bbp(image.shape[-2:], enocoded)

#     # if real_compression_ratio >= 1:
#     #     # if ssim_values["Patch HOSVD"]:
#     #     #     ssim_current = com.ssim(image, reconstructed)
#     #     #     ssim_last = ssim_values["Patch HOSVD"][-1]
#     #     #     if abs(ssim_last - ssim_current) / ssim_last <= 1e-3:
#     #     #         del compression_ratios["Patch HOSVD"][-1]
#     #     #         del bpps["Patch HOSVD"][-1]
#     #     #         del reconstructed_images["Patch HOSVD"][-1]
#     #     #         del ssim_values["Patch HOSVD"][-1]
#     #     #         del psnr_values["Patch HOSVD"][-1]

#     compression_ratios["Patch HOSVD"].append(real_compression_ratio)
#     bpps["Patch HOSVD"].append(real_bpp)
#     reconstructed_images["Patch HOSVD"].append(reconstructed)
#     psnr_values["Patch HOSVD"].append(com.psnr(image, reconstructed))
#     ssim_values["Patch HOSVD"].append(com.ssim(image, reconstructed))


selected_methods = [
    "JPEG",
    # "JPEG2000",
    # "WEBP",
    # "SVD",
    "Patch SVD",
    "Patch IMF - RGB",
    "Patch IMF - YCbCr",
    # "Patch HOSVD",
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
plt.xlim(0.05, 0.35)
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
plt.xlim(0.05, 0.35)
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
selected_bpps = [0.3, 0.2, 0.15, 0.1, 0.05]
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
