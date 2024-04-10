import os
import numpy as np
import torch
from torchvision.transforms import v2
import matplotlib.pyplot as plt

from torch.utils.data import DataLoader
from src.models import compression as com
from src.utils import CustomDataset 

script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(script_dir)

# Load an image or an image dataset. data_dir can be a path or a directory
data_dir = 'data/kodak/'
data_dir = os.path.join(parent_dir,data_dir)

# transforms = v2.Compose([v2.ToImage(), v2.Resize(size=(224, 224), interpolation=2)])
transforms = v2.Compose([v2.ToImage()])

dataset = CustomDataset(root_dir=data_dir, transform=transforms)
dataloader = DataLoader(dataset, batch_size=1, shuffle=False)


def calc_compression_metrics(image, method_list):
    """ Calculates reconstructed images and metric values for each image and each method in the method_list """

    if "Patch SVD" in method_list:
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

    if "Patch IMF" in method_list:
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


selected_methods = [
    "JPEG",
    # "JPEG2000",
    "WEBP",
    "Patch SVD",
    "Patch IMF",
    # "Patch HOSVD",
]

# initiating metrics
compression_ratios = {method: [] for method in selected_methods}
bpps = {method: [] for method in selected_methods}
reconstructed_images = {method: [] for method in selected_methods}
psnr_values = {method: [] for method in selected_methods}
ssim_values = {method: [] for method in selected_methods}


for image_id, image in enumerate(dataloader):
    print(f"processing image {image_id + 1}")
    calc_compression_metrics(image.squeeze(), method_list=selected_methods)


image_num = len(dataloader)
# Plotting the results: PSNR vs bpp
plt.figure()
for method, values in psnr_values.items():
    y_values = np.mean(np.array(values).reshape((image_num,-1)), axis=0)
    x_values = np.mean(np.array(bpps[method]).reshape((image_num,-1)), axis=0)
    plt.plot(x_values, y_values, marker="o", markersize=4, label=method)

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
    y_values = np.mean(np.array(values).reshape((image_num,-1)), axis=0)
    x_values = np.mean(np.array(bpps[method]).reshape((image_num,-1)), axis=0)
    plt.plot(x_values, y_values, marker="o", markersize=4, label=method)

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
