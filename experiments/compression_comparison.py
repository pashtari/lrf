import numpy as np
import torch
from torchvision.transforms import v2
import matplotlib.pyplot as plt
from skimage import data
from skimage.io import imread

from src.models import compression as com


# Load the astronaut image
# image = data.astronaut()
image = imread(
    "/Users/pooya/Library/CloudStorage/OneDrive-KULeuven/research/projects/lsvd/data/kodak/kodim21.png"
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
    "Interpolation": [],
    "Interpolation - Antialias": [],
    "Interpolation - Low": [],
    "DCT": [],
    "DCT - Low": [],
    "JPEG": [],
    "SVD": [],
    "Patch SVD": [],
    "Patch IMF": [],
    "HOSVD": [],
    "Patch HOSVD": [],
    "TTD": [],
    "Patch TTD": [],
}

bpps = {
    "Interpolation": [],
    "Interpolation - Antialias": [],
    "Interpolation - Low": [],
    "DCT": [],
    "DCT - Low": [],
    "JPEG": [],
    "SVD": [],
    "Patch SVD": [],
    "Patch IMF": [],
    "HOSVD": [],
    "Patch HOSVD": [],
    "TTD": [],
    "Patch TTD": [],
}

reconstructed_images = {
    "Interpolation": [],
    "Interpolation - Antialias": [],
    "Interpolation - Low": [],
    "DCT": [],
    "DCT - Low": [],
    "JPEG": [],
    "SVD": [],
    "Patch SVD": [],
    "Patch IMF": [],
    "HOSVD": [],
    "Patch HOSVD": [],
    "TTD": [],
    "Patch TTD": [],
}

psnr_values = {
    "Interpolation": [],
    "Interpolation - Antialias": [],
    "DCT": [],
    "JPEG": [],
    "SVD": [],
    "Patch SVD": [],
    "Patch IMF": [],
    "HOSVD": [],
    "Patch HOSVD": [],
    "TTD": [],
    "Patch TTD": [],
}
ssim_values = {
    "Interpolation": [],
    "Interpolation - Antialias": [],
    "DCT": [],
    "JPEG": [],
    "SVD": [],
    "Patch SVD": [],
    "Patch IMF": [],
    "HOSVD": [],
    "Patch HOSVD": [],
    "TTD": [],
    "Patch TTD": [],
}

# Calculate reconstructed images and metric values for each method

# # Interpolation
# for new_size in range(32, 224, 16):
#     interpolate = com.Interpolate(mode="bilinear", antialias=False)
#     reconstructed_interpolate = interpolate(image, new_size=new_size)
#     compression_ratios["Interpolation"].append(interpolate.real_compression_ratio)
#     reconstructed_images["Interpolation"].append(reconstructed_interpolate)
#     psnr_values["Interpolation"].append(com.psnr(image, reconstructed_interpolate))
#     ssim_values["Interpolation"].append(com.ssim(image, reconstructed_interpolate))

# # Interpolation - Antialias
# for new_size in range(32, 224, 16):
#     interpolate = com.Interpolate(mode="bilinear", antialias=True)
#     reconstructed_interpolate = interpolate(image, new_size=new_size)
#     compression_ratios["Interpolation - Antialias"].append(
#         interpolate.real_compression_ratio
#     )
#     reconstructed_images["Interpolation - Antialias"].append(reconstructed_interpolate)
#     psnr_values["Interpolation - Antialias"].append(
#         com.psnr(image, reconstructed_interpolate)
#     )
#     ssim_values["Interpolation - Antialias"].append(
#         com.ssim(image, reconstructed_interpolate)
#     )

# # Interpolation - low
# for new_size in range(32, 224, 16):
#     interpolate = com.Interpolate(mode="bilinear")
#     reconstructed_interpolate_low = interpolate.encode(
#         image, new_size=new_size
#     )
#     compression_ratios["Interpolation - Low"].append(interpolate.real_compression_ratio)
#     reconstructed_images["Interpolation - Low"].append(reconstructed_interpolate_low)

# # DCT
# for cutoff in range(32, 224, 16):
#     dct = com.DCT()
#     reconstructed_dct = dct(image, cutoff=cutoff)
#     compression_ratios["DCT"].append(dct.real_compression_ratio)
#     reconstructed_images["DCT"].append(reconstructed_dct)
#     psnr_values["DCT"].append(com.psnr(image, reconstructed_dct))
#     ssim_values["DCT"].append(com.ssim(image, reconstructed_dct))

# # DCT low
# for cutoff in range(32, 224, 16):
#     dct = com.DCT()
#     reconstructed_dct_low = com.minmax_normalize(dct(image, cutoff=cutoff, pad=False))
#     compression_ratios["DCT - Low"].append(dct.real_compression_ratio)
#     reconstructed_images["DCT - Low"].append(reconstructed_dct_low)


# JPEG
for quality in range(1, 95, 1):
    enocoded = com.jpeg_encode(image, quality=quality)
    reconstructed = com.jpeg_decode(enocoded)

    real_compression_ratio = com.get_compression_ratio(image, enocoded)
    real_bpp = com.get_bbp(image.shape[-2:], enocoded)

    compression_ratios["JPEG"].append(real_compression_ratio)
    bpps["JPEG"].append(real_bpp)
    reconstructed_images["JPEG"].append(reconstructed)
    psnr_values["JPEG"].append(com.psnr(image, reconstructed))
    ssim_values["JPEG"].append(com.ssim(image, reconstructed))


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
for quality in np.linspace(0.0, 0.2, 50):
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


# Patch IMF
for quality in np.linspace(0.0, 0.2, 50):
    enocoded = com.patch_imf_encode(
        image, quality=quality, patch_size=(8, 8), dtype=torch.int8, num_iters=10
    )
    reconstructed = com.patch_imf_decode(enocoded)

    real_compression_ratio = com.get_compression_ratio(image, enocoded)
    real_bpp = com.get_bbp(image.shape[-2:], enocoded)

    compression_ratios["Patch IMF"].append(real_compression_ratio)
    bpps["Patch IMF"].append(real_bpp)
    reconstructed_images["Patch IMF"].append(reconstructed)
    psnr_values["Patch IMF"].append(com.psnr(image, reconstructed))
    ssim_values["Patch IMF"].append(com.ssim(image, reconstructed))


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
#     enocoded = com.patch_hosvd_encode(
#         image, bpp=b, patch_size=(8, 8), dtype=torch.int16
#     )
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


# # TTD
# for rank in range(6, 224, 8):
#     ttd = com.TTD()
#     reconstructed_ttd = ttd(image, rank=(3, rank))
#     if ttd.real_compression_ratio >= 1:
#         compression_ratios["TTD"].append(ttd.real_compression_ratio)
#         reconstructed_images["TTD"].append(reconstructed_ttd)
#         psnr_values["TTD"].append(com.psnr(image, reconstructed_ttd))
#         ssim_values["TTD"].append(com.ssim(image, reconstructed_ttd))


# # Patch TTD
# for ratio in np.linspace(1, 55, 50):
#     patch_ttd = com.PatchTTD()
#     reconstructed_patch_ttd = patch_ttd(image, compression_ratio=ratio)
#     if patch_ttd.real_compression_ratio >= 1:
#         compression_ratios["Patch TTD"].append(patch_ttd.real_compression_ratio)
#         reconstructed_images["Patch TTD"].append(reconstructed_patch_ttd)
#         psnr_values["Patch TTD"].append(com.psnr(image, reconstructed_patch_ttd))
#         ssim_values["Patch TTD"].append(com.ssim(image, reconstructed_patch_ttd))


selected_methods = [
    # "Interpolation",
    # "SVD",
    # "HOSVD",
    # "TTD",
    # "DCT",
    "JPEG",
    "Patch SVD",
    "Patch IMF",
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
plt.xlim(0, 2)
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
plt.xlim(0, 2)
plt.legend()
plt.grid()
plt.savefig(
    "experiments/compression_methods_comparison_ssim-bpp.pdf",
    format="pdf",
    dpi=600,
)
plt.show()


# Plotting the compressed images for each method and bpp
selected_bpps = [1, 0.5, 0.3, 0.2, 0.15, 0.1, 0.05]
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
