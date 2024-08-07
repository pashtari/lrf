import numpy as np
import pandas as pd

import lrf


# Set directories
data = "kodak"
task = "ablation_bounds"
results_dir = f"experiments/{task}"
figures_dir = "paper/v2-tip/manuscript/figures"


# Load results
results = lrf.read_config(f"{results_dir}/{task}_results.json")
results = pd.DataFrame(results)
results = results.query("`bit rate (bpp)` < 0.8")
results["bounds"] = [str(x) for x in results["bounds"]]


# Plot PSNR vs bpp
plot = lrf.Plot(
    results, columns=("data", "method", "bit rate (bpp)", "bounds", "PSNR (dB)")
)

plot.interpolate(
    x="bit rate (bpp)",
    y="PSNR (dB)",
    x_values=np.linspace(0.05, 0.5, 19),
    groupby=["bounds", "data"],
)

plot.plot(
    x="bit rate (bpp)",
    y="PSNR (dB)",
    groupby="bounds",
    errorbar="se",
    dashed=True,
    xlim=(0.05, 0.5),
    ylim=(17, None),
    legend_labels=["[-128, 127]", "[-32, 31]", "[-16, 15]", "[-8, 7]"],
)

plot.save(save_dir=results_dir, prefix=task, format="pdf")
plot.save(save_dir=figures_dir, prefix=task, format="pgf")
