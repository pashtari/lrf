import os
import sys
import glob
import json
import numpy as np
from torch.nn.modules.utils import _pair

script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(script_dir)
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

os.chdir(parent_dir)
os.environ["PROJECT_ROOT"] = parent_dir
from utils.utils import plot_result

task_name = "ablation_iternum"
study_variable = "num_iters"
def parse_legend(x):
    """modify this function accordingly"""
    return f"\# iters = {x}"


pattern = os.path.join(script_dir, "**", f"*args.json")
files = glob.glob(pattern, recursive=True)
files.sort(key=os.path.getmtime)

run_num = len(files)
bpps = {}
psnr_values = {}
ssim_values = {}
legend = []
for file in files:
    with open(file, 'r') as f:
        args_dict = json.load(f)

    result_path = os.path.join(os.path.dirname(file), "results.json")
    with open(result_path, 'r') as f:
        results_dict = json.load(f)

    bpps[f"{parse_legend(args_dict[study_variable])}"] = results_dict["bpps"][args_dict["selected_methods"][0]]
    psnr_values[f"{parse_legend(args_dict[study_variable])}"] = results_dict["psnr_values"][args_dict["selected_methods"][0]]
    ssim_values[f"{parse_legend(args_dict[study_variable])}"] = results_dict["ssim_values"][args_dict["selected_methods"][0]]


x_axis_fixed_values = np.linspace(0.05, 0.5, 25)
# plotting the results: PSNR vs bpp
fig_data_dict = {"xlabel": "rate (bpp)", "ylabel": "PSNR (dB)", "title": " ", "xlim": (0.05,0.5), "ylim": (7,30), "fontsize": 16}
plot_result(x_values=bpps, y_values=psnr_values, x_axis_fixed_values=x_axis_fixed_values, plot_num=24, figure_data=fig_data_dict, save_dir="../../paper/figures", file_name=f"{task_name}_psnr")

# plotting the results: PSNR vs bpp
fig_data_dict["ylabel"] = "SSIM"
fig_data_dict["ylim"] = (0.15,0.9)
plot_result(x_values=bpps, y_values=ssim_values, x_axis_fixed_values=x_axis_fixed_values, plot_num=24, figure_data=fig_data_dict, save_dir="../../paper/figures", file_name=f"{task_name}_ssim")


    


    


