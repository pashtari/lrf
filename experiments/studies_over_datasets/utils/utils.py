import numpy as np
import torch
import matplotlib.pyplot as plt
from copy import deepcopy
from scipy.interpolate import interp1d
import tikzplotlib
import lrf
import os
import datetime
import pickle
from torch.nn.modules.utils import _pair


def tikzplotlib_fix_ncols(obj):
    """
    workaround for matplotlib >= 3.6 renamed legend's _ncol to _ncols, which breaks tikzplotlib
    """
    if hasattr(obj, "_ncols"):
        obj._ncol = obj._ncols
    for child in obj.get_children():
        tikzplotlib_fix_ncols(child)

def plot_result(x_values, y_values, x_axis_fixed_values, plot_num, figure_data=None, save_dir=".", file_name="results"):
    
    y_values_interpolated = {}
    for method, values in y_values.items():
        x_axis = np.array(x_values[method]).reshape((plot_num,-1))
        y_axis = np.array(values).reshape((plot_num,-1))
        x_axis_fixed_vals = deepcopy(x_axis_fixed_values)
        x_axis_fixed_vals[x_axis_fixed_vals < x_axis.min()] = None
        x_axis_fixed_vals[x_axis_fixed_vals > x_axis.max()] = None
        val_matrix = np.zeros((plot_num, len(x_axis_fixed_vals)))
        for i in range(plot_num):
            x_ax, unique_idx = np.unique(x_axis[i], return_index=True)
            y_ax = y_axis[i,unique_idx]
            interp_func = interp1d(x_ax, y_ax, kind='linear', fill_value="extrapolate")
            val_matrix[i] = interp_func(x_axis_fixed_vals)
        
        y_values_interpolated[method] = {"val_mat": val_matrix, "x_axis": x_axis_fixed_vals}

    
    mean_y_values = {method: np.mean(val_dict["val_mat"], axis=0) for method, val_dict in y_values_interpolated.items()}
    std_y_values = {method: np.std(val_dict["val_mat"], axis=0) for method, val_dict in y_values_interpolated.items()}
    
    fig = plt.figure()
    for method, val_dict in y_values_interpolated.items():
        plt.plot(val_dict["x_axis"], mean_y_values[method], marker="o", markersize=4, label=method)
        plt.fill_between(val_dict["x_axis"], mean_y_values[method] - std_y_values[method], mean_y_values[method] + std_y_values[method], alpha=0.2)

    
    plt.xlabel(figure_data["xlabel"])
    plt.ylabel(figure_data["ylabel"])
    plt.title(figure_data["title"])
    plt.xlim(figure_data["xlim"][0], figure_data["xlim"][1])

    plt.legend()
    plt.grid()

    tikzplotlib_fix_ncols(fig)
    tikzplotlib.save(f"{save_dir}/{file_name}.tex")
    plt.savefig(f"{save_dir}/{file_name}.pdf", format="pdf", dpi=600)
    plt.show()
    plt.close()



def calc_compression_metrics(dataloader, compression_ratios, bpps, psnr_values, ssim_values, args):
    """ Calculates reconstructed images and metric values for each image and each method in the method_list """

    bounds = args.bounds if isinstance(args.bounds,tuple) else (-args.bounds, args.bounds-1)

    image_num = len(dataloader)
    for image_id, image in enumerate(dataloader):
        print(f"processing image {image_id + 1}/{image_num}", flush=True)
        image = image.squeeze()

        if "JPEG" in args.selected_methods:
            for quality in range(0, 60, 1):
                enocoded = lrf.pil_encode(image, format="JPEG", quality=quality)
                reconstructed = lrf.pil_decode(enocoded)

                real_compression_ratio = lrf.get_compression_ratio(image, enocoded)
                real_bpp = lrf.get_bbp(image.shape[-2:], enocoded)

                compression_ratios["JPEG"].append(real_compression_ratio)
                bpps["JPEG"].append(real_bpp)
                # reconstructed_images["JPEG"].append(reconstructed)
                psnr_values["JPEG"].append(lrf.psnr(image, reconstructed))
                ssim_values["JPEG"].append(lrf.ssim(image, reconstructed))

        if "SVD" in args.selected_methods:
            for quality in np.linspace(0.0, 7, 30):
                enocoded = lrf.svd_encode(
                    image, quality=quality, patch=args.pachify, patch_size=args.patch_size, dtype=torch.int8
                )
                reconstructed = lrf.svd_decode(enocoded)

                real_compression_ratio = lrf.get_compression_ratio(image, enocoded)
                real_bpp = lrf.get_bbp(image.shape[-2:], enocoded)

                compression_ratios["SVD"].append(real_compression_ratio)
                bpps["SVD"].append(real_bpp)
                # reconstructed_images["SVD"].append(reconstructed)
                psnr_values["SVD"].append(lrf.psnr(image, reconstructed))
                ssim_values["SVD"].append(lrf.ssim(image, reconstructed))


        if "IMF-RGB" in args.selected_methods:
            for quality in np.linspace(0.0, 30, 50):
                enocoded = lrf.imf_encode(
                    image,
                    color_space="RGB",
                    quality=quality,
                    patch=args.pachify,
                    patch_size=_pair(args.patch_size),
                    bounds=bounds,
                    dtype=torch.int8,
                    num_iters=10,
                    verbose=False,
                )
                reconstructed = lrf.imf_decode(enocoded)

                real_compression_ratio = lrf.get_compression_ratio(image, enocoded)
                real_bpp = lrf.get_bbp(image.shape[-2:], enocoded)

                compression_ratios["IMF-RGB"].append(real_compression_ratio)
                bpps["IMF-RGB"].append(real_bpp)
                # reconstructed_images["IMF-RGB"].append(reconstructed)
                psnr_values["IMF-RGB"].append(lrf.psnr(image, reconstructed))
                ssim_values["IMF-RGB"].append(lrf.ssim(image, reconstructed))


        if "IMF-YCbCr" in args.selected_methods:
            for quality in np.linspace(0, 50, 50):
                enocoded = lrf.imf_encode(
                    image,
                    color_space="YCbCr",
                    scale_factor=(0.5, 0.5),
                    quality=(quality, quality / 2, quality / 2),
                    patch=args.pachify,
                    patch_size=_pair(args.patch_size),
                    bounds=bounds,
                    dtype=torch.int8,
                    num_iters=args.num_iters,
                    verbose=False,
                )
                reconstructed = lrf.imf_decode(enocoded)

                real_compression_ratio = lrf.get_compression_ratio(image, enocoded)
                real_bpp = lrf.get_bbp(image.shape[-2:], enocoded)

                compression_ratios["IMF-YCbCr"].append(real_compression_ratio)
                bpps["IMF-YCbCr"].append(real_bpp)
                # reconstructed_images["IMF-YCbCr"].append(reconstructed)
                psnr_values["IMF-YCbCr"].append(lrf.psnr(image, reconstructed))
                ssim_values["IMF-YCbCr"].append(lrf.ssim(image, reconstructed))

    return compression_ratios, bpps, psnr_values, ssim_values


def make_result_dir(experiment_dir, args):
    current_datetime = datetime.datetime.now()
    formatted_datetime = current_datetime.strftime('%Y-%m-%d--%H:%M:%S')
    result_dir = os.path.join(experiment_dir,f"{formatted_datetime}")
    os.makedirs(result_dir)

    with open(os.path.join(result_dir, 'args.pkl'), 'wb') as f:
        pickle.dump(args, f)

    return result_dir
