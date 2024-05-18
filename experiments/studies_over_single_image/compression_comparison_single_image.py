import os
import time
import json
import numpy as np
import torch
from torchvision.transforms import v2
import matplotlib
import matplotlib.pyplot as plt
from skimage.io import imread

import lrf

matplotlib.use("pgf")
matplotlib.rcParams.update(
    {
        "pgf.texsystem": "pdflatex",
        "font.family": "serif",
        "pgf.rcfonts": False,
    }
)

script_dir = os.path.dirname(os.path.abspath(__file__))

# Load the image
task_name = "test"
image = imread("/Users/pourya/Library/CloudStorage/git_repos/my_projects/lsvd/data/kodak/kodim19.png")
# image = imread("./data/clic/clic2024_validation_image/724b1dfdbde05257c46bb5e9863995b7c37bfcf101ed5a233ef2aa26193f09c4.png")

task_dir = os.path.join(script_dir, task_name)
if not os.path.exists(task_dir):
    os.makedirs(task_dir)

# Transform the input image
transforms = v2.Compose([v2.ToImage()])
image = torch.tensor(transforms(image))

# Visualize image
plt.imshow(image.permute(1, 2, 0))
plt.axis("off")
# plt.title("Original Image", fontsize=24)
plt.savefig(
    f"{task_dir}/{task_name}_original.pdf",
    format="pdf",
    dpi=600,
)
plt.show()
plt.close()


# Store errors and compressed images for each method and CR/bpp
compression_ratios = {
    "JPEG": [],
    "WEBP": [],
    "SVD": [],
    "IMF - RGB": [],
    "IMF": [],
}

bpps = {
    "JPEG": [],
    "WEBP": [],
    "SVD": [],
    "IMF - RGB": [],
    "IMF": [],
}

reconstructed_images = {
    "JPEG": [],
    "WEBP": [],
    "SVD": [],
    "IMF - RGB": [],
    "IMF": [],
}

psnr_values = {
    "JPEG": [],
    "WEBP": [],
    "SVD": [],
    "IMF - RGB": [],
    "IMF": [],
}
ssim_values = {
    "JPEG": [],
    "WEBP": [],
    "SVD": [],
    "IMF - RGB": [],
    "IMF": [],
}

encode_time = {
    "JPEG": [],
    "WEBP": [],
    "SVD": [],
    "IMF - RGB": [],
    "IMF": [],
}

decode_time = {
    "JPEG": [],
    "WEBP": [],
    "SVD": [],
    "IMF - RGB": [],
    "IMF": [],
}

# Calculate reconstructed images and metric values for each method


# JPEG
# for quality in range(0, 40, 1):
#     t0 = time.time()
#     enocoded = lrf.pil_encode(image, format="JPEG", quality=quality)
#     t1 = time.time()
#     reconstructed = lrf.pil_decode(enocoded)
#     t2 = time.time()

#     real_compression_ratio = lrf.get_compression_ratio(image, enocoded)
#     real_bpp = lrf.get_bbp(image.shape[-2:], enocoded)

#     compression_ratios["JPEG"].append(real_compression_ratio)
#     bpps["JPEG"].append(real_bpp)
#     reconstructed_images["JPEG"].append(reconstructed)
#     psnr_values["JPEG"].append(lrf.psnr(image, reconstructed).item())
#     ssim_values["JPEG"].append(lrf.ssim(image, reconstructed).item())
#     encode_time["JPEG"].append(1000 * (t1-t0))
#     decode_time["JPEG"].append(1000 * (t2-t1))


# # WebP
# for quality in range(0, 50, 1):
#     encoded = lrf.pil_encode(image, format="WEBP", quality=quality, alpha_quality=0)
#     reconstructed = lrf.pil_decode(encoded)

#     real_compression_ratio = lrf.get_compression_ratio(image, encoded)
#     real_bpp = lrf.get_bbp(image.shape[-2:], encoded)

#     compression_ratios["WEBP"].append(real_compression_ratio)
#     bpps["WEBP"].append(real_bpp)
#     reconstructed_images["WEBP"].append(reconstructed)
#     psnr_values["WEBP"].append(lrf.psnr(image, reconstructed).item())
#     ssim_values["WEBP"].append(lrf.ssim(image, reconstructed).item())

# SVD
for quality in np.linspace(0.0, 5, 20):
    t0 = time.time()
    enocoded = lrf.svd_encode(
        image,
        color_space="RGB",
        quality=quality,
        patch=True,
        patch_size=(8, 8),
        dtype=torch.int8,
    )
    t1 = time.time()
    reconstructed = lrf.svd_decode(enocoded)
    t2 = time.time()

    real_compression_ratio = lrf.get_compression_ratio(image, encoded)
    real_bpp = lrf.get_bbp(image.shape[-2:], encoded)

    compression_ratios["SVD"].append(real_compression_ratio)
    bpps["SVD"].append(real_bpp)
    reconstructed_images["SVD"].append(reconstructed)
    psnr_values["SVD"].append(lrf.psnr(image, reconstructed).item())
    ssim_values["SVD"].append(lrf.ssim(image, reconstructed).item())
    encode_time["SVD"].append(1000 * (t1-t0))
    decode_time["SVD"].append(1000 * (t2-t1))


# IMF - RGB
for quality in np.linspace(0.0, 25, 20):
    t0 = time.time()
    enocoded = lrf.imf_encode(
        image,
        color_space="RGB",
        quality=quality,
        patch=True,
        patch_size=(8, 8),
        bounds=(-16, 15),
        dtype=torch.int8,
        num_iters=5,
        verbose=False,
    )
    t1 = time.time()
    reconstructed = lrf.imf_decode(enocoded)
    t2 = time.time()

    real_compression_ratio = lrf.get_compression_ratio(image, encoded)
    real_bpp = lrf.get_bbp(image.shape[-2:], encoded)

    compression_ratios["IMF - RGB"].append(real_compression_ratio)
    bpps["IMF - RGB"].append(real_bpp)
    reconstructed_images["IMF - RGB"].append(reconstructed)
    psnr_values["IMF - RGB"].append(lrf.psnr(image, reconstructed).item())
    ssim_values["IMF - RGB"].append(lrf.ssim(image, reconstructed).item())
    encode_time["IMF - RGB"].append(1000 * (t1-t0))
    decode_time["IMF - RGB"].append(1000 * (t2-t1))


# IMF - YCbCr
# for quality in np.linspace(0, 25, 20):
#     t0 = time.time()
#     enocoded = lrf.imf_encode(
#         image,
#         color_space="YCbCr",
#         scale_factor=(0.5, 0.5),
#         quality=(quality, quality / 2, quality / 2),
#         patch=True,
#         patch_size=(8, 8),
#         bounds=(-16, 15),
#         dtype=torch.int8,
#         num_iters=0,
#         verbose=False,
#     )
#     t1 = time.time()
#     reconstructed = lrf.imf_decode(enocoded)
#     t2 = time.time()

#     real_compression_ratio = lrf.get_compression_ratio(image, enocoded)
#     real_bpp = lrf.get_bbp(image.shape[-2:], enocoded)

#     compression_ratios["IMF"].append(real_compression_ratio)
#     bpps["IMF"].append(real_bpp)
#     reconstructed_images["IMF"].append(reconstructed)
#     psnr_values["IMF"].append(lrf.psnr(image, reconstructed).item())
#     ssim_values["IMF"].append(lrf.ssim(image, reconstructed).item())
#     encode_time["IMF"].append(1000 * (t1-t0))
#     decode_time["IMF"].append(1000 * (t2-t1))


selected_methods = [
    # "JPEG",
    # "WEBP",
    "SVD",
    "IMF - RGB",
    # "IMF",
]
bpps = {k: bpps[k] for k in selected_methods}
reconstructed_images = {k: reconstructed_images[k] for k in selected_methods}
psnr_values = {k: psnr_values[k] for k in selected_methods}
ssim_values = {k: ssim_values[k] for k in selected_methods}
encode_time = {k: encode_time[k] for k in selected_methods}
decode_time = {k: decode_time[k] for k in selected_methods}

selected_bpps = [0.4, 0.3, 0.2, 0.15, 0.1]


def plot_metrics_collage():
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
        f"{task_dir}/compression_methods_comparison_psnr-bpp.pdf",
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
        f"{task_dir}/compression_methods_comparison_ssim-bpp.pdf",
        format="pdf",
        dpi=600,
    )
    plt.show()

    # Plotting the compressed images for each method and bpp
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
        f"{task_dir}/compression_methods_qualitative_comparison.pdf",
        format="pdf",
        dpi=600,
    )
    plt.show()


def plot_individual_figs(title=False):
    plt.figure()
    for i, bbp in enumerate(selected_bpps):
        for j, method in enumerate(selected_methods):
            ii = np.argmin(np.abs(np.array(bpps[method]) - bbp))
            reconstructed_image = reconstructed_images[method][ii]
            real_bpp = bpps[method][ii]
            plt.imshow(reconstructed_image.permute(1, 2, 0))
            if title:
                plt.title(method, fontsize=24)
            plt.axis("off")
            plt.savefig(
                f"{task_dir}/{task_name}_{method}_bpp_{real_bpp:.3f}.pdf",
                format="pdf",
                dpi=600,
            )


def save_metadata():
    metadata = {"psnr_values": psnr_values, "ssim_values":ssim_values, "bpps": bpps, "encode_time": encode_time, "decode_time": decode_time}

    with open(f"{task_dir}/{task_name}_metadata.json", "w") as json_file:
        json.dump(metadata, json_file, indent=4)

# plot_metrics_collage()
# plot_individual_figs()
save_metadata()
