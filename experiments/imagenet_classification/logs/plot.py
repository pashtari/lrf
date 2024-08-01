import glob
import re

import numpy as np
import pandas as pd

from lrf.utils import utils


def read_results(path, metrics=("bpp", "top1", "top5")):
    paths = glob.glob(path, recursive=True)
    results = []
    for path in paths:
        res = {}
        for metric in metrics:
            with open(path, "r") as log_file:
                log_string = log_file.read()
                match = re.search(
                    rf"{metric}\s*:\s*[+-]?(\d+(\.\d*)?|\.\d+)([eE][+-]?\d+)?",
                    log_string,
                )
                if match:
                    res[metric] = float(match.group(1))

        results.append(res)

    return results


#### Read and prepare results ####
data = "imagenet"
results_dir = "experiments/imagenet_classification/logs"
figures_dir = "paper/v2-tip/manuscript/figures"

# Load and collect results
results = []
for method in ("JPEG", "SVD", "IMF"):
    res = read_results(
        f"{results_dir}/**/*{data}*{method.lower()}*.log",
        ("bpp", "top1_accuracy", "top5_accuracy"),
    )
    if res:
        res = pd.DataFrame(res)
        res["method"] = method
        results.append(res)

results = pd.concat(results)

# Convert accuracy to percentage
results["top5_accuracy"] = results["top5_accuracy"] * 100
results["top1_accuracy"] = results["top1_accuracy"] * 100

# Rename metrics for compatibality
results.rename(columns={"bpp": "bit rate (bpp)"}, inplace=True)
results.rename(columns={"top1_accuracy": "top-1 accuracy (%)"}, inplace=True)
results.rename(columns={"top5_accuracy": "top-5 accuracy (%)"}, inplace=True)

# Remove rows with missing values and high bbps
results.dropna(inplace=True)
results = results.query("`bit rate (bpp)` < 0.8")

#### Plot top-1 vs bbp ####
plot = utils.Plot(results)

# Interpolation for better visualization
plot.interpolate(
    x="bit rate (bpp)",
    y="top-1 accuracy (%)",
    x_values=np.linspace(0.05, 0.5, 19),
    groupby="method",
)

# Plot and set styles
plot.plot(
    x="bit rate (bpp)",
    y="top-1 accuracy (%)",
    groupby="method",
    errorbar="se",
    dashed=True,
    xlim=(0.05, 0.5),
    ylim=(5, 80),
    legend_labels=["JPEG", "SVD", "IMF"],
)

# Save figure
plot.save(save_dir=results_dir, prefix=data, format="pdf")
plot.save(save_dir=figures_dir, prefix=data, format="pgf")


#### Plot top-5 vs bbp ####
plot = utils.Plot(results)

# Interpolate for better visualization
plot.interpolate(
    x="bit rate (bpp)",
    y="top-5 accuracy (%)",
    x_values=np.linspace(0.05, 0.5, 19),
    groupby="method",
)

# Plot and set styles
plot.plot(
    x="bit rate (bpp)",
    y="top-5 accuracy (%)",
    groupby="method",
    errorbar="se",
    dashed=True,
    xlim=(0.05, 0.5),
    ylim=(5, 95),
    legend_labels=["JPEG", "SVD", "IMF"],
)

# Save figure
plot.save(save_dir=results_dir, prefix=data, format="pdf")
plot.save(save_dir=figures_dir, prefix=data, format="pgf")
