import os
import glob
import pickle
import numpy as np
from torch.nn.modules.utils import _pair
from utils import plot_result

script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(script_dir)

task_name = "ablation_patchsize"
study_variable = "patch_size"
def parse_legend(x):
    """modify this function accordingly"""
    return f"patchsize {_pair(x)}"

result_dir = os.path.join(parent_dir, task_name)

pattern = os.path.join(result_dir, "**", f"*args.pkl")
files = glob.glob(pattern, recursive=True)

run_num = len(files)
bpps = {}
psnr_values = {}
ssim_values = {}
legend = []
for file in files:
    with open(file, 'rb') as f:
        args_dict = pickle.load(f)

    result_path = os.path.join(os.path.dirname(file), "resutls.pkl")
    with open(result_path, 'rb') as f:
        results_dict = pickle.load(f)

    bpps[f"{args_dict.selected_methods[0]} - {parse_legend(args_dict.__dict__[study_variable])}"] = results_dict["bpps"][args_dict.selected_methods[0]]
    psnr_values[f"{args_dict.selected_methods[0]} - {parse_legend(args_dict.__dict__[study_variable])}"] = results_dict["psnr_values"][args_dict.selected_methods[0]]
    ssim_values[f"{args_dict.selected_methods[0]} - {parse_legend(args_dict.__dict__[study_variable])}"] = results_dict["ssim_values"][args_dict.selected_methods[0]]

x_axis_fixed_values = np.linspace(0, 1, 50)
# plotting the results: PSNR vs bpp
fig_data_dict = {"xlabel": "bpp", "ylabel": "PSNR (dB)", "title": "Ablation study over Kodak", "xlim": (0,1)}
plot_result(x_values=bpps, y_values=psnr_values, x_axis_fixed_values=x_axis_fixed_values, plot_num=24, figure_data=fig_data_dict, save_dir=result_dir, file_name=f"{task_name}_psnr")

# plotting the results: PSNR vs bpp
fig_data_dict["ylabel"] = "SSIM"
plot_result(x_values=bpps, y_values=ssim_values, x_axis_fixed_values=x_axis_fixed_values, plot_num=24, figure_data=fig_data_dict, save_dir=result_dir, file_name=f"{task_name}_ssim")


    


    


