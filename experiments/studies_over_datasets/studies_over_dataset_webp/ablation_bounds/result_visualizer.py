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

os.chdir(parent_dir)
os.environ["PROJECT_ROOT"] = parent_dir
from utils.utils import plot_result, save_metadata

task_name = "ablation_bounds"
study_variable = "bounds"
def parse_legend(x):
    """modify this function accordingly"""
    return f"bound ({-x}, {x-1})"


pattern = os.path.join(script_dir, "**", f"*args.pkl")
files = glob.glob(pattern, recursive=True)

run_num = len(files)
bpps = {}
psnr_values = {}
ssim_values = {}
legend = []
for i in [4,0,1,5]:
    file = files[i]
    with open(file, 'rb') as f:
        args_dict = pickle.load(f)

    result_path = os.path.join(os.path.dirname(file), "results.pkl")
    with open(result_path, 'rb') as f:
        results_dict = pickle.load(f)

    bpps[f"{parse_legend(args_dict.__dict__[study_variable])}"] = results_dict["bpps"][args_dict.selected_methods[0]]
    psnr_values[f"{parse_legend(args_dict.__dict__[study_variable])}"] = results_dict["psnr_values"][args_dict.selected_methods[0]]
    ssim_values[f"{parse_legend(args_dict.__dict__[study_variable])}"] = results_dict["ssim_values"][args_dict.selected_methods[0]]

x_axis_fixed_values = np.linspace(0.05, 0.5, 25)
# plotting the results: PSNR vs bpp
fig_data_dict = {"xlabel": "bpp", "ylabel": "PSNR (dB)", "title": "", "xlim": (0.05,0.5), "ylim": (20,30), "fontsize": 16}
plot_result(x_values=bpps, y_values=psnr_values, x_axis_fixed_values=x_axis_fixed_values, plot_num=24, figure_data=fig_data_dict, save_dir=script_dir, file_name=f"{task_name}_psnr")

# plotting the results: PSNR vs bpp
fig_data_dict["ylabel"] = "SSIM"
fig_data_dict["ylim"] = (0.4, 0.85)
plot_result(x_values=bpps, y_values=ssim_values, x_axis_fixed_values=x_axis_fixed_values, plot_num=24, figure_data=fig_data_dict, save_dir=script_dir, file_name=f"{task_name}_ssim")

metadata = {"bpps": bpps, "psnr_values": psnr_values, "ssim_values": ssim_values}
save_metadata(metadata=metadata, save_dir=script_dir)

    


    


