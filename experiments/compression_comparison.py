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
    compression_ratios = np.array([1.25, 1.5, 1.75, *np.arange(2, 30.5, 2)])

    # Store errors and compressed images for each method and compression ratio
    compression_ratios = {
        "Interpolation": [],
        "Interpolation - Antialias": [],
        "Interpolation - Low": [],
        "DCT": [],
        "DCT - Low": [],
        # "SVD": [],
        "Patch SVD": [],
    }
    err_values = {
        "Interpolation": [],
        "Interpolation - Antialias": [],
        "DCT": [],
        # "SVD": [],
        "Patch SVD": [],
    }
    compressed_images = {
        "Interpolation": [],
        "Interpolation - Antialias": [],
        "Interpolation - Low": [],
        "DCT": [],
        "DCT - Low": [],
        # "SVD": [],
        "Patch SVD": [],
    }

    # Calculate reconstructed images and metric for each method and compression ratio
    # Interpolation
    for new_size in range(48, 224, 16):
        interpolate = com.Interpolate(mode="bilinear", antialias=False)
        compressed_interpolate = interpolate(image, new_size=new_size).clip(0, 1)
        compression_ratios["Interpolation"].append(interpolate.real_compression_ratio)
        err_values["Interpolation"].append(metric(image, compressed_interpolate))
        compressed_images["Interpolation"].append(compressed_interpolate)

    # Interpolation - Antialias
    for new_size in range(48, 224, 16):
        interpolate = com.Interpolate(mode="bilinear", antialias=True)
        compressed_interpolate = interpolate(image, new_size=new_size).clip(0, 1)
        compression_ratios["Interpolation - Antialias"].append(
            interpolate.real_compression_ratio
        )
        err_values["Interpolation - Antialias"].append(
            metric(image, compressed_interpolate)
        )
        compressed_images["Interpolation - Antialias"].append(compressed_interpolate)

    # Interpolation - low
    for new_size in range(48, 224, 16):
        interpolate = com.Interpolate(mode="bilinear")
        compressed_interpolate_low = interpolate.compress(image, new_size=new_size).clip(
            0, 1
        )
        compression_ratios["Interpolation - Low"].append(
            interpolate.real_compression_ratio
        )
        compressed_images["Interpolation - Low"].append(compressed_interpolate_low)

    # DCT
    for cutoff in range(48, 224, 16):
        dct = com.DCT()
        compressed_dct = dct(image, cutoff=cutoff).clip(0, 1)
        compression_ratios["DCT"].append(dct.real_compression_ratio)
        err_values["DCT"].append(metric(image, compressed_dct))
        compressed_images["DCT"].append(compressed_dct)

    # DCT low
    for cutoff in range(48, 224, 16):
        dct = com.DCT()
        compressed_dct_low = com.minmax_normalize(dct(image, cutoff=cutoff, pad=False))
        compression_ratios["DCT - Low"].append(dct.real_compression_ratio)
        compressed_images["DCT - Low"].append(com.minmax_normalize(compressed_dct_low))

        # # SVD
        # svd = com.SVD()
        # compressed_svd = svd(image, ratio).clip(0, 1)
        # err_values["SVD"].append(metric(image, compressed_svd))
        # compressed_images["SVD"].append(compressed_svd)

    # Patch SVD
    for rank in range(7, 192, 16):
        patch_svd = com.PatchSVD(patch_size=(8, 8))
        compressed_patch_svd = patch_svd(image, rank=rank).clip(0, 1)
        compression_ratios["Patch SVD"].append(patch_svd.real_compression_ratio)
        err_values["Patch SVD"].append(metric(image, compressed_patch_svd))
        compressed_images["Patch SVD"].append(compressed_patch_svd)

    # Plotting the results
    plt.figure()
    for method, err in err_values.items():
        plt.plot(compression_ratios[method], err, marker="o", label=method)

    plt.xlabel("Compression Ratio")
    plt.ylabel(f"{metric.__name__.upper()}")
    plt.title("Comprison of Different Compression Methods")
    # plt.xticks(np.arange(1, compression_ratios.max() + 1))
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
    len(compression_ratios["Patch SVD"]),
    len(compressed_images),
    figsize=(5 * len(err_values), 3 * len(compression_ratios["Patch SVD"])),
)

# Setting titles for columns
for ax, method in zip(axs[0], compressed_images.keys()):
    ax.set_title(method)

# Setting labels for rows
for ax, ratio in zip(axs[:, 0], compression_ratios):
    ax.set_ylabel(f"Ratio {ratio}", rotation=90, size="large")

for i, ratio in enumerate(compression_ratios["Patch SVD"]):
    for j, method in enumerate(compressed_images.keys()):
        ii = np.argmax(
            np.where(
                ratio > np.array(compression_ratios[method]),
                np.array(compression_ratios[method]),
                -np.inf,
            )
        )
        compressed_image = compressed_images[method][ii]
        axs[i, j].imshow(compressed_image.squeeze(0).permute(1, 2, 0))
        axs[i, j].axis("off")


plt.tight_layout()
plt.savefig(
    "experiments/compression_methods_qualitative_comparison.pdf",
    format="pdf",
    dpi=600,
)
plt.show()
