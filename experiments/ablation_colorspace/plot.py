import numpy as np
import pandas as pd

import lrf


# Set directories
data = "kodak"
task = "ablation_colorspace"
results_dir = f"experiments/{task}"
figures_dir = "paper/v2-tip/manuscript/figures"


# Load results
results = lrf.read_config(f"{results_dir}/{task}_results.json")
results = pd.DataFrame(results)
results = results.query("`bit rate (bpp)` < 0.8")
results.rename(columns={"color_space": "color space"}, inplace=True)


# Plot PSNR vs bpp
plot = lrf.Plot(
    results, columns=("data", "method", "bit rate (bpp)", "color space", "PSNR (dB)")
)

plot.interpolate(
    x="bit rate (bpp)",
    y="PSNR (dB)",
    x_values=np.linspace(0.05, 0.5, 19),
    groupby=["color space", "data"],
)

plot.plot(
    x="bit rate (bpp)",
    y="PSNR (dB)",
    groupby="color space",
    errorbar="se",
    dashed=True,
    xlim=(0.05, 0.5),
    legend_labels=["RGB", "YCbCr"],
)

plot.save(save_dir=results_dir, prefix=task, format="pdf")
plot.save(save_dir=figures_dir, prefix=task, format="pgf")
