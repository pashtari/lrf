import numpy as np
import matplotlib.pyplot as plt

from src.utils import get_eval_results
from src.models import compression as com


plt.rcParams.update({"font.size": 12})


size = 224
fig, ax = plt.subplots(figsize=(9, 7))
max_com_ratio = 20


# Interpolation
new_size, top1_acc = get_eval_results(
    "logs/eval_interpolate_imagenet_pretrained_resnet50",
    filename=None,
    extension=".log",
    x_par_name="model.new_size",
    y_par_name="val_top1_accuracy",
    config_rel_path=".hydra/overrides.yaml",
)

interpolate = com.Interpolate()
com_ratios = np.array([interpolate.get_compression_ratio(size, t) for t in new_size])
mask = (com_ratios <= max_com_ratio) & ~np.isnan(top1_acc)
com_ratios = com_ratios[mask]
top1_acc = top1_acc[mask]
com_ratios, top1_acc = zip(*sorted(zip(com_ratios, top1_acc)))

ax.plot(com_ratios, top1_acc, marker="o", label="Interpolation")


# Interpolation - low
new_size, top1_acc = get_eval_results(
    "logs/eval_interpolate_imagenet_pretrained_resnet50_low",
    filename=None,
    extension=".log",
    x_par_name="model.new_size",
    y_par_name="val_top1_accuracy",
    config_rel_path=".hydra/overrides.yaml",
)

interpolate = com.Interpolate()
com_ratios = np.array([interpolate.get_compression_ratio(size, t) for t in new_size])
mask = (com_ratios <= max_com_ratio) & ~np.isnan(top1_acc)
com_ratios = com_ratios[mask]
top1_acc = top1_acc[mask]
com_ratios, top1_acc = zip(*sorted(zip(com_ratios, top1_acc)))

ax.plot(com_ratios, top1_acc, marker="o", label="Interpolation - Low")


# DCT
new_size, top1_acc = get_eval_results(
    "logs/eval_dct_imagenet_pretrained_resnet50",
    filename=None,
    extension=".log",
    x_par_name="model.new_size",
    y_par_name="val_top1_accuracy",
    config_rel_path=".hydra/overrides.yaml",
)

dct = com.DCT()
com_ratios = np.array([dct.get_compression_ratio(size, t) for t in new_size])
mask = (com_ratios <= max_com_ratio) & ~np.isnan(top1_acc)
com_ratios = com_ratios[mask]
top1_acc = top1_acc[mask]
com_ratios, top1_acc = zip(*sorted(zip(com_ratios, top1_acc)))

ax.plot(com_ratios, top1_acc, marker="o", label="DCT")


# DCT - low
new_size, top1_acc = get_eval_results(
    "logs/eval_dct_imagenet_pretrained_resnet50_low",
    filename=None,
    extension=".log",
    x_par_name="model.new_size",
    y_par_name="val_top1_accuracy",
    config_rel_path=".hydra/overrides.yaml",
)

dct = com.DCT()
com_ratios = np.array([dct.get_compression_ratio(size, t) for t in new_size])
mask = (com_ratios <= max_com_ratio) & ~np.isnan(top1_acc)
com_ratios = com_ratios[mask]
top1_acc = top1_acc[mask]
com_ratios, top1_acc = zip(*sorted(zip(com_ratios, top1_acc)))

ax.plot(com_ratios, top1_acc, marker="o", label="DCT - Low")


# PatchSVD
rank, top1_acc = get_eval_results(
    "logs/eval_patchsvd_imagenet_pretrained_resnet50",
    filename=None,
    extension=".log",
    x_par_name="model.rank",
    y_par_name="val_top1_accuracy",
    config_rel_path=".hydra/overrides.yaml",
)

patch_size = 8
patchsvd = com.PatchSVD(patch_size)
patchified_size = ((size / patch_size) ** 2, 3 * patch_size**2)
com_ratios = np.array([patchsvd.get_compression_ratio(patchified_size, r) for r in rank])
mask = (com_ratios <= max_com_ratio) & ~np.isnan(top1_acc)
com_ratios = com_ratios[mask]
top1_acc = top1_acc[mask]
com_ratios, top1_acc = zip(*sorted(zip(com_ratios, top1_acc)))

ax.plot(com_ratios, top1_acc, marker="o", label="Patch SVD")


# Save plot
ax.grid()
ax.set_xlabel("Compression Ratio")
ax.set_ylabel("Top1 Accuracy")
ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.15), ncol=5)
# fig.subplots_adjust(bottom=0.2)
fig.savefig(
    "experiments/compression_methods_comparison_top1_acc.pdf",
    format="pdf",
    dpi=600,
)
