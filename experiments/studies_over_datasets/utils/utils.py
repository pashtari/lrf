import os
import json
import time
import datetime
import pickle
import matplotlib
import matplotlib.pyplot as plt
import tikzplotlib
import numpy as np
from scipy.interpolate import interp1d
import torch
from torch.nn.modules.utils import _pair

import lrf

matplotlib.use("pgf")
plt.rcParams.update({
    "font.family": "serif",  # use serif/main font for text elements
    "text.usetex": True,     # use inline math for ticks
    "pgf.rcfonts": False     # don't setup fonts from rc parameters
    })

def plot_result(x_values, y_values, x_axis_fixed_values, plot_num, legend_list=None, figure_data=None, save_dir=".", file_name="results"):
    
    y_values_interpolated = {}
    for method, values in y_values.items():
        x_axis = np.array(x_values[method]).reshape((plot_num,-1))
        y_axis = np.array(values).reshape((plot_num,-1))

        x_axis_min = x_axis.min()
        x_axis_fixed_values_ = x_axis_fixed_values[x_axis_fixed_values>=x_axis_min]

        val_matrix = np.zeros((plot_num, len(x_axis_fixed_values_)))
        for i in range(plot_num):
            x_ax, unique_idx = np.unique(x_axis[i], return_index=True)
            y_ax = y_axis[i,unique_idx]
            interp_func = interp1d(x_ax, y_ax, kind='linear', fill_value='extrapolate')
            val_matrix[i] = interp_func(x_axis_fixed_values_)
        
        y_values_interpolated[method] = {"val_mat": val_matrix, "x_axis_min": x_axis_min}

    
    mean_y_values = {method: np.mean(val_dict["val_mat"], axis=0) for method, val_dict in y_values_interpolated.items()}
    std_y_values = {method: np.std(val_dict["val_mat"], axis=0) for method, val_dict in y_values_interpolated.items()}
    
    fig = plt.figure()
    fixed_value_step = x_axis_fixed_values[1] - x_axis_fixed_values[0]
    colors = ['C0', 'C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8']
    for i, (method, dict) in enumerate(y_values_interpolated.items()):
        plt.plot(x_axis_fixed_values[x_axis_fixed_values >= dict["x_axis_min"]], mean_y_values[method], "o-", color=colors[i], markersize=6, linewidth=3, label=method if legend_list is None else legend_list[i])
        shade_minus = mean_y_values[method] - std_y_values[method] 
        shade_plus = mean_y_values[method] + std_y_values[method]
        plt.fill_between(x_axis_fixed_values[x_axis_fixed_values >= dict["x_axis_min"]], shade_minus, shade_plus, alpha=0.2, color=colors[i])
        
        interp_func_for_mean = interp1d(x_axis_fixed_values[x_axis_fixed_values >= dict["x_axis_min"]], mean_y_values[method], kind='linear', fill_value='extrapolate')
        exptapolated_mean = interp_func_for_mean(x_axis_fixed_values[x_axis_fixed_values <= dict["x_axis_min"]+fixed_value_step])
        plt.plot(x_axis_fixed_values[x_axis_fixed_values <= dict["x_axis_min"]+fixed_value_step], exptapolated_mean, "o--", color=colors[i], markersize=6, linewidth=3)
    
    fontsize = figure_data["fontsize"] if "fontsize" in figure_data else 10

    plt.yticks(fontsize=fontsize)
    plt.xticks(fontsize=fontsize)

    plt.xlabel(figure_data["xlabel"], fontsize=fontsize)
    plt.ylabel(figure_data["ylabel"], fontsize=fontsize)
    plt.title(figure_data["title"], fontsize=fontsize)
    plt.xlim(figure_data["xlim"][0], figure_data["xlim"][1])
    if "ylim" in figure_data.keys():
        plt.ylim(figure_data["ylim"][0], figure_data["ylim"][1])

    plt.legend(loc='lower right', fontsize=fontsize-2)
    plt.grid()

    plt.savefig(f"{save_dir}/{file_name}.pgf", format="pgf", dpi=600)
    plt.savefig(f"{save_dir}/{file_name}.pdf", format="pdf", dpi=600)
    # tikzplotlib.save(f"{save_dir}/{file_name}.tex")
    plt.show()
    plt.close()


def calc_compression_metrics(dataloader, compression_ratios, bpps, psnr_values, ssim_values, elapsed_times_encode, elapsed_times_decode, args):
    """ Calculates reconstructed images and metric values for each image and each method in the method_list """

    bounds = args.bounds if isinstance(args.bounds,tuple) else (-args.bounds, args.bounds-1)

    image_num = len(dataloader)
    for image_id, (image, _) in enumerate(dataloader):
        print(f"processing image {image_id + 1}/{image_num}", flush=True)
        image = image.squeeze()

        if "JPEG" in args.selected_methods:
            for quality in range(0, 60, 1):
                t0 = time.time()
                enocoded = lrf.pil_encode(image, format="JPEG", quality=quality)
                t1 = time.time()
                reconstructed = lrf.pil_decode(enocoded)
                t2 = time.time()

                real_compression_ratio = lrf.get_compression_ratio(image, enocoded)
                real_bpp = lrf.get_bbp(image.shape[-2:], enocoded)

                compression_ratios["JPEG"].append(real_compression_ratio)
                bpps["JPEG"].append(real_bpp)
                # reconstructed_images["JPEG"].append(reconstructed)
                psnr_values["JPEG"].append(lrf.psnr(image, reconstructed).item())
                ssim_values["JPEG"].append(lrf.ssim(image, reconstructed).item())
                elapsed_times_encode["JPEG"].append(1000*(t1 - t0))
                elapsed_times_decode["JPEG"].append(1000*(t2 - t1))

        if "SVD" in args.selected_methods:
            for quality in np.linspace(0.0, 7, 30):
                t0 = time.time()
                enocoded = lrf.svd_encode(
                    image, quality=quality, patch=args.patchify, patch_size=_pair(args.patch_size), dtype=torch.int8
                )
                t1 = time.time()
                reconstructed = lrf.svd_decode(enocoded)
                t2 = time.time()

                real_compression_ratio = lrf.get_compression_ratio(image, enocoded)
                real_bpp = lrf.get_bbp(image.shape[-2:], enocoded)

                compression_ratios["SVD"].append(real_compression_ratio)
                bpps["SVD"].append(real_bpp)
                # reconstructed_images["SVD"].append(reconstructed)
                psnr_values["SVD"].append(lrf.psnr(image, reconstructed).item())
                ssim_values["SVD"].append(lrf.ssim(image, reconstructed).item())
                elapsed_times_encode["SVD"].append(1000*(t1 - t0))
                elapsed_times_decode["SVD"].append(1000*(t2 - t1))


        if "IMF" in args.selected_methods:
            for quality in np.linspace(0, 50, 50):
                in_quality = (quality, quality / 2, quality / 2) if args.color_space == "YCbCr" else quality
                t0 = time.time()
                enocoded = lrf.imf_encode(
                    image,
                    color_space=args.color_space,
                    scale_factor=(0.5, 0.5),
                    quality=in_quality,
                    patch=args.patchify,
                    patch_size=_pair(args.patch_size),
                    bounds=bounds,
                    dtype=torch.int8,
                    num_iters=args.num_iters,
                    verbose=False,
                )
                t1 = time.time()
                reconstructed = lrf.imf_decode(enocoded)
                t2 = time.time()

                real_compression_ratio = lrf.get_compression_ratio(image, enocoded)
                real_bpp = lrf.get_bbp(image.shape[-2:], enocoded)

                compression_ratios["IMF"].append(real_compression_ratio)
                bpps["IMF"].append(real_bpp)
                # reconstructed_images["IMF"].append(reconstructed)
                psnr_values["IMF"].append(lrf.psnr(image, reconstructed).item())
                ssim_values["IMF"].append(lrf.ssim(image, reconstructed).item())
                elapsed_times_encode["IMF"].append(1000*(t1 - t0))
                elapsed_times_decode["IMF"].append(1000*(t2 - t1))

    return compression_ratios, bpps, psnr_values, ssim_values, elapsed_times_encode, elapsed_times_decode


def make_result_dir(experiment_dir, args):
    current_datetime = datetime.datetime.now()
    formatted_datetime = current_datetime.strftime('%Y-%m-%d--%H:%M:%S')
    result_dir = os.path.join(experiment_dir,f"{formatted_datetime}")
    os.makedirs(result_dir)

    with open(os.path.join(result_dir, 'args.json'), "w") as f:
        json.dump(args.__dict__, f, indent=4)

    return result_dir

def save_metadata(metadata, save_dir="."):
    with open(f"{save_dir}/metadata.json", 'w') as json_file:
        json.dump(metadata, json_file, indent=4)