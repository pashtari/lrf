import os
import pickle
import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

from pathlib import Path
from skimage import img_as_float
from torchvision.transforms import Resize
from torch.utils.data import Subset

from src.models import compression as com

# Set current script path as the working directory
cwd = Path(__file__).parent
os.chdir(cwd)

# Load dataset
task_name = "compression_comparison_CIFAR10"
dataset = torchvision.datasets.CIFAR10(root='../datasets', train=False, download=True, transform=transforms.ToTensor())
dataset = Subset(dataset, indices=[0, 1, 2, 3])
data_loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False)

methods = [com.compress_interpolate, com.compress_dct, com.compress_patch_svd]
metrics = [com.psnr, com.ssim]
compression_ratios = np.array([1.25, *np.arange(2, 25.5, 1)])

metric_values = {method.__name__: {metric.__name__: np.zeros((len(compression_ratios), 1)) for metric in metrics} for method in methods}
metric_values["image_count"] = 0
metric_values["image_max_num"] = len(data_loader)

resize_img = Resize((224,224), antialias=True)
# Loop over the dataset
for img_counter, (image_orig, label) in enumerate(data_loader):
    # Convert the image to float and resize
    image = resize_img(image_orig)
    image = image.squeeze().permute(1,2,0)
    image = img_as_float(image)
    # image = exposure.equalize_hist(image)

    running_ave_coef = img_counter/(img_counter+1)
    metric_values["image_count"] = img_counter
    print(f"processing image {img_counter+1}/{len(data_loader)}")

    for metric in metrics:
        for i, ratio in enumerate(compression_ratios):
            for method in methods:
                compressed_image = method(image, ratio)
                metric_values[method.__name__][metric.__name__][i] = running_ave_coef * metric_values[method.__name__][metric.__name__][i] + (1 - running_ave_coef) * metric(image, compressed_image)

# saving the dictionary
with open(f'{task_name}_dict.pkl', 'wb') as f:
    pickle.dump(metric_values, f)

# Plotting the results
for metric in metrics:
    plt.figure()
    for method in methods:
        plt.plot(compression_ratios, metric_values[method.__name__][metric.__name__], marker="o", label=method.__name__)

    plt.xlabel("Compression Ratio")
    plt.ylabel(f"{metric.__name__.upper()}")
    plt.title("Comprison of Different Compression Methods")
    plt.xticks(np.arange(1, compression_ratios.max() + 1))
    plt.legend()
    plt.grid()
    plt.savefig(
        f"{task_name}_{metric.__name__}.pdf",
        format="pdf",
        dpi=600,
    )
    plt.show()
