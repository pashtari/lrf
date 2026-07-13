"""Evaluate ResNet-50 on the compressed ImageNet validation set.

One invocation = one operating point of Figure 5: compress every validation
image with the given method and quality, run standard ResNet-50 inference,
and record (method, quality, mean bpp, top-1, top-5) as one row of the
results CSV. Points already in the CSV are skipped, so sweeps are resumable.

Example:
    python src/evaluate.py --method qmf --quality 5.0
"""

import argparse
import csv
import os

import torch
from torch.utils.data import DataLoader
from torchvision import models
from torchvision.datasets import ImageFolder
from torchvision.transforms.v2 import functional as F
from tqdm import tqdm

import qmf

SRC_DIR = os.path.dirname(os.path.abspath(__file__))
PACKAGE_ROOT = os.path.dirname(SRC_DIR)
# The dataset lives in the repository's data folder, next to the other datasets
REPO_DATA_DIR = os.path.normpath(os.path.join(PACKAGE_ROOT, "..", "..", "data"))
WNIDS_FILE = os.path.join(SRC_DIR, "imagenet_wnids.txt")
CSV_FIELDS = ["method", "quality", "bpp", "top1_accuracy", "top5_accuracy"]

# Preprocessing constants of the experiment (do not change: they define Figure 5)
RESIZE_SIZE, CROP_SIZE = 256, 224
IMAGE_MEAN, IMAGE_STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]

CODECS = {  # method -> (encode, decode); encoders take only `quality`, rest are qmf defaults
    "qmf": (qmf.qmf_encode, qmf.qmf_decode),
    "jpeg": (lambda image, quality: qmf.pil_encode(image, format="JPEG", quality=quality), qmf.pil_decode),
    "svd": (qmf.svd_encode, qmf.svd_decode),
}


def build_transform(method, quality):
    """PIL image -> (preprocessed tensor, bpp).

    Compression runs at the ORIGINAL resolution and bpp is measured there
    (compressed bits / original pixel count); resize/crop/normalize follow.
    Method "none" is the uncompressed baseline (raw 8-bit RGB = 24 bpp).
    """
    if method == "jpeg":
        if quality != int(quality):
            raise ValueError(f"JPEG quality must be an integer, got {quality}")
        quality = int(quality)  # PIL rejects float quality

    def transform(pil_image):
        image = F.pil_to_tensor(pil_image)  # uint8, CHW, RGB
        if method == "none":
            bpp = 24.0
        else:
            encode, decode = CODECS[method]
            encoded = encode(image, quality=quality)
            bpp = qmf.bits_per_pixel(image.shape[-2:], encoded)
            image = decode(encoded)
        image = F.resize(image, RESIZE_SIZE, interpolation=F.InterpolationMode.BILINEAR, antialias=True)
        image = F.center_crop(image, CROP_SIZE)
        image = F.to_dtype(image, torch.float32, scale=True)
        image = F.normalize(image, IMAGE_MEAN, IMAGE_STD)
        return image, torch.tensor(bpp)

    return transform


class ImageNetVal(ImageFolder):
    """ImageNet val in the layout <root>/[val/]<wnid>/*.JPEG.

    Labels use the canonical ILSVRC2012/torchvision convention — the index of
    the wnid in the sorted list of all 1000 wnids (imagenet_wnids.txt) — so
    pretrained models need no remapping. Any SUBSET of classes works too
    (handy for quick sanity checks), which a plain ImageFolder would mislabel.
    """

    def __init__(self, root, transform=None):
        val_dir = os.path.join(root, "val")
        super().__init__(val_dir if os.path.isdir(val_dir) else root, transform=transform)

    def find_classes(self, directory):
        with open(WNIDS_FILE) as f:
            canonical = {wnid: i for i, wnid in enumerate(f.read().split())}
        classes = sorted(e.name for e in os.scandir(directory) if e.is_dir() and e.name in canonical)
        if not classes:
            raise FileNotFoundError(
                f"No ImageNet wnid subdirectories (e.g. n01440764) found in {directory}. "
                "Run src/download_imagenet_val.py or point --imagenet-dir at a prepared copy."
            )
        return classes, {c: canonical[c] for c in classes}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", required=True, choices=["qmf", "jpeg", "svd", "none"])
    parser.add_argument("--quality", type=float, required=True, help="compression quality")
    parser.add_argument(
        "--imagenet-dir",
        default=os.path.join(REPO_DATA_DIR, "imagenet"),
        help="ImageNet root: <dir>/[val/]<wnid>/*.JPEG (default: ../../data/imagenet)",
    )
    parser.add_argument(
        "--results",
        default=os.path.join(PACKAGE_ROOT, "results", "results.csv"),
        help="results CSV (default: ./results/results.csv)",
    )
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument(
        "--num-workers", type=int, default=8,
        help="dataloader workers; compression runs here on the CPU, so more cores = faster",
    )
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu",
        help="inference device (default: cuda when available)",
    )
    parser.add_argument("--force", action="store_true", help="re-run a point, replacing its CSV row")
    return parser.parse_args()


def read_rows(results_csv):
    if not os.path.isfile(results_csv) or os.path.getsize(results_csv) == 0:
        return []
    with open(results_csv, newline="") as f:
        return list(csv.DictReader(f))


def is_same_point(row, method, quality):
    return row["method"].lower() == method and abs(float(row["quality"]) - quality) < 1e-9


@torch.no_grad()
def evaluate(args):
    dataset = ImageNetVal(args.imagenet_dir, transform=build_transform(args.method, args.quality))
    loader = DataLoader(
        dataset, batch_size=args.batch_size, num_workers=args.num_workers,
        pin_memory=True, shuffle=False,
    )
    weights = models.ResNet50_Weights.IMAGENET1K_V2
    model = models.resnet50(weights=weights).eval().to(args.device)
    print(f"ResNet-50 ({weights}) on {args.device}; {len(dataset)} images")

    top1 = top5 = 0
    bpp_sum = 0.0
    for (images, bpps), labels in tqdm(loader, desc=f"{args.method} quality={args.quality:g}"):
        logits = model(images.to(args.device, non_blocking=True))
        preds = logits.topk(5, dim=1).indices.cpu()
        top1 += (preds[:, 0] == labels).sum().item()
        top5 += (preds == labels[:, None]).any(dim=1).sum().item()
        bpp_sum += bpps.sum().item()

    n = len(dataset)
    return {
        "method": args.method.upper(),
        "quality": f"{args.quality:g}",
        "bpp": f"{bpp_sum / n:.5f}",
        "top1_accuracy": f"{top1 / n:.5f}",
        "top5_accuracy": f"{top5 / n:.5f}",
    }


def main():
    args = parse_args()
    rows = read_rows(args.results)
    if not args.force and any(is_same_point(r, args.method, args.quality) for r in rows):
        print(f"{args.method} quality={args.quality:g} already in {args.results} — skipping")
        return

    row = evaluate(args)
    rows = [r for r in rows if not is_same_point(r, args.method, args.quality)] + [row]
    os.makedirs(os.path.dirname(args.results) or ".", exist_ok=True)
    with open(args.results, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    print(
        f"{row['method']} quality={row['quality']} -> bpp {row['bpp']}, "
        f"top-1 {row['top1_accuracy']}, top-5 {row['top5_accuracy']} (saved to {args.results})"
    )


if __name__ == "__main__":
    main()
