import os
import glob
import argparse

import numpy as np
import torch

import lrf


def get_args():
    parser = argparse.ArgumentParser(description="Ablation study on QMF patch size.")
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
        default="ablation_patchsize",
        help="save dir (default: ablation_patchsize)",
    )
    parser.add_argument(
        "--prefix",
        type=str,
        default="ablation_patchsize",
        help="prefix for saved files (default: ablation_patchsize)",
    )

    args = parser.parse_args()

    if args.data_dir is None:
        args.data_dir = f"../data/{args.data}"

    return args


def eval_image(image):
    image_id = os.path.basename(image)
    image = lrf.read_image(image)

    results = []

    for patch_size, patch in [
        (4, True),
        (8, True),
        (16, True),
        (32, True),
        (None, False),
    ]:
        for quality in np.linspace(0, 40, 80):
            params = {
                "color_space": "YCbCr",
                "scale_factor": (0.5, 0.5),
                "quality": (quality, quality / 2, quality / 2),
                "patch": patch,
                "patch_size": (patch_size, patch_size),
                "bounds": (-16, 15),
                "dtype": torch.int8,
                "num_iters": 10,
                "verbose": False,
            }
            config = {"data": image_id, "method": "QMF", **params}
            log = lrf.eval_compression(image, lrf.qmf_encode, lrf.qmf_decode, **params)
            results.append({**config, **log})

        print(
            f"method QMF, patch size {(patch_size, patch_size)}, image {config['data']} done."
        )

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
