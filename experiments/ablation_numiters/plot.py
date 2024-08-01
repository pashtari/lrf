import numpy as np
import pandas as pd

from lrf.utils import utils


# Set directories
data = "kodak"
task = "ablation_numiters"
results_dir = f"experiments/{task}"
figures_dir = "paper/v2-tip/manuscript/figures"


# Load results
results = utils.read_config(f"{results_dir}/{task}_results.json")
results = pd.DataFrame(results)
results = results.query("`bit rate (bpp)` < 0.8")
results.rename(columns={"num_iters": r"\# iterations"}, inplace=True)
results[r"\# iterations"] = [str(x) for x in results[r"\# iterations"]]
# results = results[results["\# iterations"] != "0"]


# Plot PSNR vs bpp
plot = utils.Plot(
    results, columns=("data", "method", "bit rate (bpp)", r"\# iterations", "PSNR (dB)")
)

plot.interpolate(
    x="bit rate (bpp)",
    y="PSNR (dB)",
    x_values=np.linspace(0.05, 0.5, 19),
    groupby=[r"\# iterations", "data"],
)

plot.plot(
    x="bit rate (bpp)",
    y="PSNR (dB)",
    groupby=r"\# iterations",
    errorbar="se",
    dashed=True,
    xlim=(0.05, 0.5),
    ylim=(6.25, 29.25),
    legend_labels=["0", "1", "2", "5", "10"],
)

plot.save(save_dir=results_dir, prefix=task, format="pdf")
plot.save(save_dir=figures_dir, prefix=task, format="pgf")
