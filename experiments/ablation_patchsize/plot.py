import numpy as np
import pandas as pd

from lrf.utils import utils

# Set directories
data = "kodak"
task = "ablation_patchsize"
results_dir = f"experiments/{task}"
figures_dir = "paper/v2-tip/manuscript/figures"


# Load results
results = utils.read_config(f"{results_dir}/{task}_results.json")
results = pd.DataFrame(results)
results = results.query("`bit rate (bpp)` < 0.8")
results["patch size"] = [
    "no patchification" if None in x else str(tuple(x)) for x in results["patch_size"]
]

# Plot PSNR vs bpp
plot = utils.Plot(
    results, columns=("data", "method", "bit rate (bpp)", "patch size", "PSNR (dB)")
)

plot.interpolate(
    x="bit rate (bpp)",
    y="PSNR (dB)",
    x_values=np.linspace(0.05, 0.5, 19),
    groupby=["patch size", "data"],
)

plot.plot(
    x="bit rate (bpp)",
    y="PSNR (dB)",
    groupby="patch size",
    errorbar="se",
    dashed=True,
    xlim=(0.05, 0.5),
    legend_labels=["(4, 4)", "(8, 8)", "(16, 16)", "(32, 32)", "no patchification"],
)

plot.save(save_dir=results_dir, prefix=task, format="pdf")
plot.save(save_dir=figures_dir, prefix=task, format="pgf")
