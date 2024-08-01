import os
import glob
import argparse

import numpy as np
import torch

import lrf
from lrf.utils import utils


def get_args():
    parser = argparse.ArgumentParser(
        description="Compare compression methods over a dataset."
    )
    parser.add_argument(
        "--data", type=str, default="kodak", help="dataset name (default: kodak)"
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        nargs="?",
        help="path to dataset (default: ../data/{data})",
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        default="comparison",
        help="save dir (default: comparison)",
    )
    parser.add_argument(
        "--prefix", type=str, nargs="?", help="prefix for saved files (default: {data})"
    )

    args = parser.parse_args()

    if args.data_dir is None:
        args.data_dir = f"../data/{args.data}"

    if args.prefix is None:
        args.prefix = args.data

    return args


def eval_image(image):
    image_id = os.path.basename(image)
    image = utils.read_image(image)

    results = []

    # JPEG
    for quality in range(0, 75, 1):
        params = {"quality": quality}
        config = {"data": image_id, "method": "JPEG", **params}
        log = utils.eval_compression(
            image,
            lrf.pil_encode,
            lrf.pil_decode,
            format="JPEG",
            **params,
        )
        results.append({**config, **log})
        print(
            f"method {config['method']}, quality {quality}, image {config['data']} done."
        )

    # SVD
    for quality in np.linspace(0.0, 5, 30):
        params = {
            "color_space": "RGB",
            "quality": quality,
            "patch": True,
            "patch_size": (8, 8),
        }
        config = {"data": image_id, "method": "SVD", **params}
        log = utils.eval_compression(image, lrf.svd_encode, lrf.svd_decode, **params)
        results.append({**config, **log})
        print(
            f"method {config['method']}, quality {quality}, image {config['data']} done."
        )

    # IMF - YCbCr
    for quality in np.linspace(0, 40, 80):
        params = {
            "color_space": "YCbCr",
            "scale_factor": (0.5, 0.5),
            "quality": (quality, quality / 2, quality / 2),
            "patch": True,
            "patch_size": (8, 8),
            "bounds": (-16, 15),
            "dtype": torch.int8,
            "num_iters": 10,
            "verbose": False,
        }
        config = {"data": image_id, "method": "IMF", **params}
        log = utils.eval_compression(image, lrf.imf_encode, lrf.imf_decode, **params)
        results.append({**config, **log})
        print(
            f"method {config['method']}, quality {quality}, image {config['data']} done."
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
    utils.save_config(results, save_dir=args.save_dir, prefix=args.prefix)
