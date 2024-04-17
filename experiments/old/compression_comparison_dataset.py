import os
import sys
from datetime import datetime
from pathlib import Path

import pickle
import numpy as np
import torch
from torch.utils.data import Subset
import torchvision
import torchvision.transforms as transforms
from torchvision.transforms import Resize
import matplotlib.pyplot as plt


script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(script_dir)
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

os.chdir(parent_dir)

from src.models import compression as com


# Setup result directory
now = datetime.now().strftime("%Y%m%d-%H%M%S")
folder_name = f"{now}"
output_path = Path(script_dir) / "compression_comparison_results" / folder_name
if not output_path.exists():
    output_path.mkdir(parents=True)
output_path = output_path.as_posix()

# Load dataset
task_name = "compression_comparison_CIFAR10"
dataset = torchvision.datasets.CIFAR10(
    root="./data",
    train=True,
    download=False,
    transform=transforms.ToTensor(),
)
# task_name = "compression_comparison_ImageNet"
# dataset = torchvision.datasets.ImageNet(
#     root=f"{parent_dir}/datasets", split="val", transform=transforms.ToTensor()
# )

dataset = Subset(dataset, indices=range(100))
data_loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False)


methods = [com.Interpolate(mode="bilinear"), com.DCT(), com.PatchSVD(patch_size=(4, 4))]
metrics = [com.psnr, com.ssim]
summary_metrics = [np.mean, np.std]
compression_ratios = np.array([1.25, *np.arange(2, 26, 1)])

metric_values = {
    method.__class__.__name__: {
        f"ratio_id_{i}": {
            "ratio_val": None,
            **{metric.__name__: [] for metric in metrics},
        }
        for i, _ in enumerate(compression_ratios)
    }
    for method in methods
}

metric_values["image_count"] = 0
metric_values["image_max_num"] = len(data_loader)

# Loop over the dataset
for img_counter, (image, label) in enumerate(data_loader):
    running_ave_coef = img_counter / (img_counter + 1)
    metric_values["image_count"] = img_counter
    print(f"processing image {img_counter+1}/{len(data_loader)}")

    for metric in metrics:
        for i, ratio in enumerate(compression_ratios):
            for method in methods:
                compressed_image = method(image, ratio).clip(0, 1)

                metric_values[method.__class__.__name__][f"ratio_id_{i}"][
                    metric.__name__
                ].append(metric(image, compressed_image).item())
                metric_values[method.__class__.__name__][f"ratio_id_{i}"][
                    "ratio_val"
                ] = method.real_compression_ratio


summary_values = {
    method.__class__.__name__: {
        f"ratio_id_{i}": {
            "ratio_val": None,
            **{
                metric.__name__: {summary.__name__: None for summary in summary_metrics}
                for metric in metrics
            },
        }
        for i, _ in enumerate(compression_ratios)
    }
    for method in methods
}

for metric in metrics:
    for i, _ in enumerate(compression_ratios):
        for method in methods:
            for summary in summary_metrics:
                summary_values[method.__class__.__name__][f"ratio_id_{i}"][
                    metric.__name__
                ][summary.__name__] = summary(
                    metric_values[method.__class__.__name__][f"ratio_id_{i}"][
                        metric.__name__
                    ]
                )
                summary_values[method.__class__.__name__][f"ratio_id_{i}"][
                    "ratio_val"
                ] = metric_values[method.__class__.__name__][f"ratio_id_{i}"]["ratio_val"]


# saving the dictionaries
with open(f"{output_path}/{task_name}_dict.pkl", "wb") as f:
    pickle.dump(metric_values, f)

with open(f"{output_path}/{task_name}_summary_dict.pkl", "wb") as f:
    pickle.dump(summary_values, f)

# Plotting the results
for metric in metrics:
    plt.figure()
    for method in methods:
        x_axis = np.zeros((len(compression_ratios)))
        y_axis = np.zeros((len(compression_ratios)))
        for i, _ in enumerate(compression_ratios):
            x_axis[i] = summary_values[method.__class__.__name__][f"ratio_id_{i}"][
                "ratio_val"
            ]
            y_axis[i] = summary_values[method.__class__.__name__][f"ratio_id_{i}"][
                metric.__name__
            ][summary_metrics[0].__name__]

        plt.plot(x_axis, y_axis, marker="o", label=method.__class__.__name__)

    plt.xlabel("Compression Ratio")
    plt.ylabel(f"{metric.__name__.upper()}")
    plt.title("Comprison of Different Compression Methods")
    plt.legend()
    plt.grid()
    plt.savefig(
        f"{output_path}/{task_name}_{metric.__name__}.pdf",
        format="pdf",
        dpi=600,
    )
    plt.show()
