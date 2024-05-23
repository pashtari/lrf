import os
import numpy as np
import pickle
import matplotlib.pyplot as plt
import torch
from torchvision.transforms import v2

import lrf
from lrf import CustomDataset
from torch.utils.data import DataLoader


experiment_name = "clic"
data_dir = "data/clic/clic2024_test_image"

script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(script_dir)
experiment_dir = os.path.join(script_dir, experiment_name)
if not os.path.exists(experiment_dir):
    os.makedirs(experiment_dir)

# Load an image or an image dataset. data_dir can be a path or a directory
data_dir = os.path.join(parent_dir, data_dir)

# transforms = v2.Compose([v2.ToImage(), v2.Resize(size=(224, 224), interpolation=2)])
transforms = v2.Compose([v2.ToImage()])

dataset = CustomDataset(root_dir=data_dir, transform=transforms)
dataloader = DataLoader(dataset, batch_size=1, shuffle=False)


def calc_compression_metrics(image):
    """Calculates reconstructed images and metric values for each image and each method in the method_list"""

    # JPEG
    for quality in range(0, 100, 5):
        encoded = lrf.pil_encode(image, format="JPEG", quality=quality)
        reconstructed = lrf.pil_decode(encoded)

        real_compression_ratio = lrf.get_compression_ratio(image, encoded)
        real_bpp = lrf.get_bbp(image.shape[-2:], encoded)

        compression_ratios["JPEG"].append(real_compression_ratio)
        bpps["JPEG"].append(real_bpp)
        # reconstructed_images["JPEG"].append(reconstructed)
        psnr_values["JPEG"].append(lrf.psnr(image, reconstructed))
        ssim_values["JPEG"].append(lrf.ssim(image, reconstructed))

    # SVD
    for quality in np.linspace(0.0, 100, 100):
        encoded = lrf.svd_encode(
            image, quality=quality, patch=True, patch_size=(8, 8), dtype=torch.int8
        )
        reconstructed = lrf.svd_decode(encoded)

        real_compression_ratio = lrf.get_compression_ratio(image, encoded)
        real_bpp = lrf.get_bbp(image.shape[-2:], encoded)

        compression_ratios["SVD"].append(real_compression_ratio)
        bpps["SVD"].append(real_bpp)
        # reconstructed_images["SVD"].append(reconstructed)
        psnr_values["SVD"].append(lrf.psnr(image, reconstructed))
        ssim_values["SVD"].append(lrf.ssim(image, reconstructed))

    # IMF - RGB
    for quality in np.linspace(0.0, 100, 100):
        encoded = lrf.imf_encode(
            image,
            color_space="RGB",
            quality=quality,
            patch=True,
            patch_size=(8, 8),
            bounds=(-16, 15),
            dtype=torch.int8,
            num_iters=1,
            verbose=False,
        )
        reconstructed = lrf.imf_decode(encoded)

        real_compression_ratio = lrf.get_compression_ratio(image, encoded)
        real_bpp = lrf.get_bbp(image.shape[-2:], encoded)

        compression_ratios["IMF - RGB"].append(real_compression_ratio)
        bpps["IMF - RGB"].append(real_bpp)
        # reconstructed_images["IMF - RGB"].append(reconstructed)
        psnr_values["IMF - RGB"].append(lrf.psnr(image, reconstructed))
        ssim_values["IMF - RGB"].append(lrf.ssim(image, reconstructed))

    # IMF - YCbCr
    for quality in np.linspace(0, 100, 100):
        encoded = lrf.imf_encode(
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
        reconstructed = lrf.imf_decode(encoded)

        real_compression_ratio = lrf.get_compression_ratio(image, encoded)
        real_bpp = lrf.get_bbp(image.shape[-2:], encoded)

        compression_ratios["IMF - YCbCr"].append(real_compression_ratio)
        bpps["IMF - YCbCr"].append(real_bpp)
        # reconstructed_images["IMF - YCbCr"].append(reconstructed)
        psnr_values["IMF - YCbCr"].append(lrf.psnr(image, reconstructed))
        ssim_values["IMF - YCbCr"].append(lrf.ssim(image, reconstructed))


selected_methods = [
    "JPEG",
    "SVD",
    "IMF - RGB",
    "IMF - YCbCr",
]

# initiating metrics
compression_ratios = {method: [] for method in selected_methods}
bpps = {method: [] for method in selected_methods}
# reconstructed_images = {method: [] for method in selected_methods}
psnr_values = {method: [] for method in selected_methods}
ssim_values = {method: [] for method in selected_methods}


for image_id, image in enumerate(dataloader):
    print(f"processing image {image_id + 1}/{len(dataloader)}")
    calc_compression_metrics(image=image.squeeze())


image_num = len(dataloader)
# Plotting the results: PSNR vs bpp
plt.figure()
for method, values in psnr_values.items():
    y_values = np.mean(np.array(values).reshape((image_num, -1)), axis=0)
    x_values = np.mean(np.array(bpps[method]).reshape((image_num, -1)), axis=0)
    plt.plot(x_values, y_values, marker="o", markersize=4, label=method)

plt.xlabel("bpp")
plt.ylabel("PSNR (dB)")
plt.title("Comprison of Different Compression Methods")
# plt.xticks(np.arange(1, 13, 1))
plt.xlim(0, 2)
plt.legend()
plt.grid()
plt.savefig(
    f"{experiment_dir}/compression_methods_comparison_psnr-bpp.pdf",
    format="pdf",
    dpi=600,
)
plt.show()


# Plotting the results: SSIM vs bpp
plt.figure()
for method, values in ssim_values.items():
    y_values = np.mean(np.array(values).reshape((image_num, -1)), axis=0)
    x_values = np.mean(np.array(bpps[method]).reshape((image_num, -1)), axis=0)
    plt.plot(x_values, y_values, marker="o", markersize=4, label=method)

plt.xlabel("bpp")
plt.ylabel("SSIM")
plt.title("Comprison of Different Compression Methods")
# plt.xticks(np.arange(1, 13, 1))
plt.xlim(0, 2)
plt.legend()
plt.grid()
plt.savefig(
    f"{experiment_dir}/compression_methods_comparison_ssim-bpp.pdf",
    format="pdf",
    dpi=600,
)
plt.show()

# saving the results
save_dict = {
    "compression_ratios": compression_ratios,
    "bpps": bpps,
    "psnr_values": psnr_values,
    "ssim_values": ssim_values,
}
with open(os.path.join(experiment_dir, "resutls.pkl"), "wb") as f:
    pickle.dump(save_dict, f)
