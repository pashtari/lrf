import os
import glob
import argparse

import numpy as np
import torch

import lrf


def get_args():
    parser = argparse.ArgumentParser(
        description="Ablation study on QMF bounds parameters."
    )
    parser.add_argument(
        "--data", type=str, default="kodak", help="dataset name (default: kodak)"
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        nargs="?",
        help="path to dataset (default: data/{data})",
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        default="ablation_bounds",
        help="save dir (default: ablation_bounds)",
    )
    parser.add_argument(
        "--prefix",
        type=str,
        default="ablation_bounds",
        help="prefix for saved files (default: ablation_bounds)",
    )

    args = parser.parse_args()

    if args.data_dir is None:
        args.data_dir = f"../data/{args.data}"

    return args


def eval_image(image):
    image_id = os.path.basename(image)
    image = lrf.read_image(image)

    results = []

    for bounds in [(-8, 7), (-16, 15), (-32, 31), (-128, 127)]:
        for quality in np.linspace(0, 40, 80):
            params = {
                "color_space": "YCbCr",
                "scale_factor": (0.5, 0.5),
                "quality": (quality, quality / 2, quality / 2),
                "patch": True,
                "patch_size": (8, 8),
                "bounds": bounds,
                "dtype": torch.int8,
                "num_iters": 10,
                "verbose": False,
            }
            config = {"data": image_id, "method": "QMF", **params}
            log = lrf.eval_compression(image, lrf.qmf_encode, lrf.qmf_decode, **params)
            results.append({**config, **log})

        print(f"method QMF, bounds {bounds}, image {image_id} done.")

    return results


def eval_dataset(data_dir):
    results = []
    for image in sorted(glob.glob(os.path.join(data_dir, "*.png"))):
        results.extend(eval_image(image))

    return results


if __name__ == "__main__":
    args = get_args()
    results = eval_dataset(args.data_dir)
    lrf.save_config(results, save_dir=args.save_dir, prefix=args.prefix)
