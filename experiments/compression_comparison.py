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

for metric in [com.psnr, com.ssim]:
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
        interpolate = com.Interpolate(mode="bilinear")
        compressed_interpolate = interpolate(image, ratio).clip(0, 1)
        err_values["Interpolation"].append(metric(image, compressed_interpolate))
        compressed_images["Interpolation"].append(compressed_interpolate)

        # Interpolate low
        interpolate = com.Interpolate(mode="bilinear")
        compressed_interpolate_low = interpolate.compress(image, ratio).clip(0, 1)
        compressed_images["Interpolation Low"].append(compressed_interpolate_low)

        # DCT
        dct = com.DCT()
        compressed_dct = dct(image, ratio).clip(0, 1)
        err_values["DCT"].append(metric(image, compressed_dct))
        compressed_images["DCT"].append(compressed_dct)

        # DCT Low
        dct = com.DCT()
        compressed_dct_low = com.minmax_normalize(dct(image, ratio, pad=False))
        compressed_images["DCT Low"].append(compressed_dct_low)

        # # SVD
        # svd = com.SVD()
        # compressed_svd = svd(image, ratio).clip(0, 1)
        # err_values["SVD"].append(metric(image, compressed_svd))
        # compressed_images["SVD"].append(compressed_svd)

        # SVD Patches
        patch_svd = com.PatchSVD()
        compressed_patch_svd = patch_svd(image, ratio).clip(0, 1)
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
        axs[i, j].imshow(compressed_image.squeeze(0).permute(1, 2, 0))
        axs[i, j].axis("off")


plt.tight_layout()
plt.savefig(
    "experiments/compression_methods_qualitative_comparison.pdf",
    format="pdf",
    dpi=600,
)
plt.show()
