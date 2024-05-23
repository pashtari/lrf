import os
import sys
import glob
import pickle
import numpy as np
from torch.nn.modules.utils import _pair

script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(script_dir)
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

parent_parent_dir = os.path.dirname(parent_dir)
if parent_parent_dir not in sys.path:
    sys.path.append(parent_parent_dir)

from src.utils import get_eval_results
from studies_over_datasets.utils import plot_result

task_name = "classification_performance_top1"
study_variable = "val_top1_accuracy"
method_list = ["JPEG_imagenet_classification", "SVD_imagenet_classification", "IMF_imagenet_subset_classification"]

bpps = {}
accuracies = {}
for method in method_list:
    bpps[method], accuracies[method] = get_eval_results(
        root_dir=script_dir,
        subdir=method,
        extension=".log",
        x_par_name="val_real_bpp",
        y_par_name=study_variable,
        config_rel_path=method+".log",
    )


# plotting the results
x_axis_fixed_values = np.linspace(0.05, 0.5, 25)
legend_list = ["JPEG", "SVD", "IMF"]
fig_data_dict = {"xlabel": "rate (bpp)", "ylabel": "top-1 accuracy", "title": " ", "xlim": (0.05,0.5), "ylim": (0, 0.8), "fontsize": 16}
plot_result(x_values=bpps, y_values=accuracies, x_axis_fixed_values=x_axis_fixed_values, plot_num=1, legend_list=legend_list, figure_data=fig_data_dict, save_dir=script_dir, file_name=task_name)

print("end of script")
