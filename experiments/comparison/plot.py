import numpy as np
import pandas as pd

import lrf


# Set directories
data = "clic2024"
results_dir = "experiments/comparison"
figures_dir = "paper/v2-tip/manuscript/figures"


# Load results
results = lrf.read_config(f"{results_dir}/{data}_results.json")
results = pd.DataFrame(results)
results = results.query("`bit rate (bpp)` < 0.8")

# Plot PSNR vs bpp
plot = lrf.Plot(results, columns=("data", "method", "bit rate (bpp)", "PSNR (dB)"))

plot.interpolate(
    x="bit rate (bpp)",
    y="PSNR (dB)",
    x_values=np.linspace(0.05, 0.5, 19),
    groupby=["method", "data"],
)

plot.plot(
    x="bit rate (bpp)",
    y="PSNR (dB)",
    groupby="method",
    errorbar="se",
    dashed=True,
    xlim=(0.05, 0.5),
    # ylim=(15, None),  # kodak
    ylim=(16.5, 31),  # clic
    legend_labels=["JPEG", "SVD", "QMF"],
)

plot.save(save_dir=results_dir, prefix=data, format="pdf")
plot.save(save_dir=figures_dir, prefix=data, format="pgf")


# Plot SSIM vs bpp
plot = lrf.Plot(results, columns=("data", "method", "bit rate (bpp)", "SSIM"))

plot.interpolate(
    x="bit rate (bpp)",
    y="SSIM",
    x_values=np.linspace(0.05, 0.5, 19),
    groupby=["method", "data"],
)

plot.plot(
    x="bit rate (bpp)",
    y="SSIM",
    groupby="method",
    errorbar="se",
    dashed=True,
    xlim=(0.05, 0.5),
    # ylim=(0.35, None),  # kodak
    ylim=(0.42, None),  # clic
    legend_labels=["JPEG", "SVD", "QMF"],
)

plot.save(save_dir=results_dir, prefix=data, format="pdf")
plot.save(save_dir=figures_dir, prefix=data, format="pgf")


# Plot encoding time vs bpp
plot = lrf.Plot(
    results, columns=("data", "method", "bit rate (bpp)", "encoding time (ms)")
)

plot.interpolate(
    x="bit rate (bpp)",
    y="encoding time (ms)",
    x_values=np.linspace(0.05, 0.5, 19),
    groupby=["method", "data"],
)

plot.plot(
    x="bit rate (bpp)",
    y="encoding time (ms)",
    groupby="method",
    errorbar="se",
    dashed=True,
    legend_labels=["JPEG", "SVD", "QMF"],
)

plot.save(save_dir=results_dir, prefix=data, format="pdf")


# Plot decoding time vs bpp
plot = lrf.Plot(
    results, columns=("data", "method", "bit rate (bpp)", "decoding time (ms)")
)

plot.interpolate(
    x="bit rate (bpp)",
    y="decoding time (ms)",
    x_values=np.linspace(0.05, 0.5, 19),
    groupby=["method", "data"],
)

plot.plot(
    x="bit rate (bpp)",
    y="decoding time (ms)",
    groupby="method",
    errorbar="se",
    dashed=True,
    legend_labels=["JPEG", "SVD", "QMF"],
)

plot.save(save_dir=results_dir, prefix=data, format="pdf")
