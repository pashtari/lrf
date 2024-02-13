import math

import numpy as np
import torch
import matplotlib.pyplot as plt
from skimage import data, img_as_float
from skimage import exposure
from skimage.transform import resize
from skimage.metrics import structural_similarity

from src.models import compression as com


def zscore_normalize(tensor):
    mean = np.mean(dim=(-2, -1), keepdims=True)
    std = np.std(dim=(-2, -1), keepdims=True)
    normalized_tensor = (tensor - mean) / (std + 1e-8)
    return normalized_tensor


def minmax_normalize(x):
    min_val = torch.amin(x, dim=(-2, -1), keepdim=True)
    max_val = torch.amax(x, dim=(-2, -1), keepdim=True)
    normalized_tensor = (x - min_val) / (max_val - min_val)
    return normalized_tensor


def mae(image1, image2):
    return np.mean(np.abs(image1 - image2))


def mse(image1, image2):
    return np.mean((image1 - image2) ** 2)


def psnr(image1, image2):
    mse_value = mse(image1, image2)
    if mse_value == 0:
        return float("inf")  # If the MSE is zero, the PSNR is infinite
    max_pixel = np.max(image1)
    psnr_value = 20 * math.log10(max_pixel / math.sqrt(mse_value))
    return psnr_value


def ssim(image1, image2):
    ssim_value = structural_similarity(
        image1, image2, data_range=image2.max() - image2.min(), channel_axis=-1
    )
    return ssim_value


def compress_interpolate(x, compression_ratio):
    """Compress the image by resizing using interpolation."""
    x = torch.tensor(x, dtype=torch.float32).permute(-1, 0, 1).unsqueeze(0)
    interpolate = com.Interpolate(mode="bilinear")
    y = interpolate(x, compression_ratio=compression_ratio)
    return torch.clip(y, 0, 1).squeeze(0).permute(1, 2, 0).to(torch.float64).numpy()


def compress_interpolate_low(x, compression_ratio):
    """Compress the image by resizing using interpolation."""
    x = torch.tensor(x, dtype=torch.float32).permute(-1, 0, 1).unsqueeze(0)
    interpolate = com.Interpolate(mode="bilinear")
    y = interpolate.compress(x, compression_ratio=compression_ratio)
    return torch.clip(y, 0, 1).squeeze(0).permute(1, 2, 0).to(torch.float64).numpy()


def compress_dct(x, compression_ratio):
    """Compress the image using DCT."""
    x = torch.tensor(x, dtype=torch.float32).permute(-1, 0, 1).unsqueeze(0)
    dct2 = com.DCT()
    y = dct2(x, compression_ratio=compression_ratio)
    return torch.clip(y, 0, 1).squeeze(0).permute(1, 2, 0).to(torch.float64).numpy()


def compress_dct_low(x, compression_ratio):
    """Compress the image using DCT."""
    x = torch.tensor(x, dtype=torch.float32).permute(-1, 0, 1).unsqueeze(0)
    dct2 = com.DCT()
    y = dct2(x, compression_ratio=compression_ratio, pad=False)
    return minmax_normalize(y).squeeze(0).permute(1, 2, 0).to(torch.float64).numpy()


def compress_svd(x, compression_ratio):
    """Compress the image using SVD."""
    x = torch.tensor(x, dtype=torch.float32).permute(-1, 0, 1).unsqueeze(0)
    svd = com.SVD()
    y = svd(x, compression_ratio=compression_ratio)
    return torch.clip(y, 0, 1).squeeze(0).permute(1, 2, 0).to(torch.float64).numpy()


def compress_patch_svd(x, compression_ratio, patch_size=(8, 8)):
    """Compress the image using SVD on patches."""
    x = torch.tensor(x, dtype=torch.float32).permute(-1, 0, 1).unsqueeze(0)
    patch_svd = com.PatchSVD(patch_size=patch_size)
    y = patch_svd(x, compression_ratio=compression_ratio)
    return torch.clip(y, 0, 1).squeeze(0).permute(1, 2, 0).to(torch.float64).numpy()


# Load the astronaut image
image = data.cat()

# Convert the image to float and resize
image = img_as_float(image)
image = resize(image, (224, 224), preserve_range=True)
image = exposure.equalize_hist(image)

plt.imshow(image)
plt.axis("off")
plt.title("Astronaut Image")
plt.show()

for metric in [psnr, ssim, mae]:
    # Compression ratios to be used
    compression_ratios = np.array([1.25, 1.5, 1.75, *np.arange(2, 25.5, 1)])

    # Store errors and compressed images for each method and compression ratio
    err_values = {
        "Interpolation": [],
        "DCT": [],
        # "SVD": [],
        "Patch SVD": [],
    }
    compressed_images = {
        "Interpolation": [],
        "Interpolation Low": [],
        "DCT": [],
        "DCT Low": [],
        # "SVD": [],
        "Patch SVD": [],
    }

    # Calculate reconstructed images and metric for each method and compression ratio
    for ratio in compression_ratios:
        # Interpolate
        compressed_interpolate = compress_interpolate(image, ratio)
        err_values["Interpolation"].append(metric(image, compressed_interpolate))
        compressed_images["Interpolation"].append(compressed_interpolate)

        # Interpolate low
        compressed_interpolate_low = compress_interpolate(image, ratio)
        compressed_images["Interpolation Low"].append(compressed_interpolate_low)

        # DCT
        compressed_dct = compress_dct(image, ratio)
        err_values["DCT"].append(metric(image, compressed_dct))
        compressed_images["DCT"].append(compressed_dct)

        # DCT Low
        compressed_dct_low = compress_dct_low(image, ratio)
        compressed_images["DCT Low"].append(compressed_dct_low)

        # # SVD
        # compressed_svd = compress_svd(image, ratio)
        # err_values["SVD"].append(metric(image, compressed_svd))
        # compressed_images["SVD"].append(compressed_svd)

        # SVD Patches
        compressed_patch_svd = compress_patch_svd(image, ratio)
        err_values["Patch SVD"].append(metric(image, compressed_patch_svd))
        compressed_images["Patch SVD"].append(compressed_patch_svd)

    # Plotting the results
    plt.figure()
    for method, err in err_values.items():
        plt.plot(compression_ratios, err, marker="o", label=method)

    plt.xlabel("Compression Ratio")
    plt.ylabel(f"{metric.__name__.upper()}")
    plt.title("Comprison of Different Compression Methods")
    plt.xticks(np.arange(1, compression_ratios.max() + 1))
    plt.legend()
    plt.grid()
    plt.savefig(
        f"experiments/compression_methods_comparison_{metric.__name__}.pdf",
        format="pdf",
        dpi=600,
    )
    plt.show()


# Plotting the compressed images for each method and compression ratio
fig, axs = plt.subplots(
    len(compression_ratios),
    len(compressed_images),
    figsize=(5 * len(err_values), 3 * len(compression_ratios)),
)

# Setting titles for columns
for ax, method in zip(axs[0], compressed_images.keys()):
    ax.set_title(method)

# Setting labels for rows
for ax, ratio in zip(axs[:, 0], compression_ratios):
    ax.set_ylabel(f"Ratio {ratio}", rotation=90, size="large")

for i, ratio in enumerate(compression_ratios):
    for j, method in enumerate(compressed_images.keys()):
        compressed_image = compressed_images[method][i]
        axs[i, j].imshow(compressed_image)
        axs[i, j].axis("off")

plt.tight_layout()
plt.savefig(
    "experiments/compression_methods_qualitative_comparison.pdf",
    format="pdf",
    dpi=600,
)
plt.show()
