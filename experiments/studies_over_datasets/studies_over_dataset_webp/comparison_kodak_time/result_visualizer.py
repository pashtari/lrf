import os
import sys
import glob
import pickle
import numpy as np

script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(script_dir)
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

os.chdir(parent_dir)
os.environ["PROJECT_ROOT"] = parent_dir
from utils.utils import plot_result

task_name = "comparison_kodak_time"

pattern = os.path.join(script_dir, "**", f"*results.pkl")
result_path = glob.glob(pattern, recursive=True)[0]

with open(result_path, 'rb') as f:
    results_dict = pickle.load(f)

x_axis_fixed_values = np.linspace(0.15, 0.5, 20)
# plotting the results: PSNR vs bpp
fig_data_dict = {"xlabel": "PSNR", "ylabel": "encode time (msec)", "title": " ", "xlim": (0.15, 0.5), "fontsize": 16}
plot_result(x_values=results_dict["bpps"], y_values=results_dict["elapsed_times_encode"], x_axis_fixed_values=x_axis_fixed_values, plot_num=24, figure_data=fig_data_dict, save_dir=script_dir, file_name=f"{task_name}_encode")

# plotting the results: SSIM vs bpp
fig_data_dict["ylabel"] = "decode time (msec)"
# fig_data_dict["ylim"] = (0.2,0.9)
plot_result(x_values=results_dict["bpps"], y_values=results_dict["elapsed_times_decode"], x_axis_fixed_values=x_axis_fixed_values, plot_num=24, figure_data=fig_data_dict, save_dir=script_dir, file_name=f"{task_name}_decode")