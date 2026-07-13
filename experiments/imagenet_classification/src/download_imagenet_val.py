"""Download and prepare the ImageNet-1k validation set (50,000 images).

Idempotent: if a prepared dataset already exists at --root (1000 wnid class
directories with 50 images each), the script prints a message and exits.
Otherwise it downloads the validation split from the public Hugging Face
mirror `mrm8488/ImageNet1K-val` (14 parquet shards, ~6.3 GB, original-
resolution JPEGs), extracts them into <root>/val/<wnid>/*.JPEG, verifies the
result, and removes the parquet cache.

If you already have the validation set extracted in the wnid-folder layout,
you do not need this script — pass its location via IMAGENET_DIR instead.

NOTE: ImageNet is distributed under its own terms of access
(https://image-net.org/download.php); download only if you agree to them.
"""

import argparse
import glob
import os
import shutil
import sys
from collections import Counter

WNIDS_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "imagenet_wnids.txt")
HF_REPO = "mrm8488/ImageNet1K-val"
NUM_IMAGES, NUM_CLASSES = 50000, 1000


def is_prepared(root):
    """True if root already holds a complete val set (with or without val/)."""
    val_dir = os.path.join(root, "val")
    if not os.path.isdir(val_dir):
        val_dir = root
    if not os.path.isdir(val_dir):
        return False
    class_dirs = [e for e in os.scandir(val_dir) if e.is_dir() and e.name.startswith("n")]
    if len(class_dirs) < NUM_CLASSES:
        return False
    return sum(len(os.listdir(d.path)) for d in class_dirs) >= NUM_IMAGES


def download_and_extract(root):
    import pyarrow.parquet as pq
    from huggingface_hub import snapshot_download

    with open(WNIDS_FILE) as f:
        wnids = f.read().split()

    parquet_dir = os.path.join(root, "_parquet")
    print(f"Downloading {HF_REPO} (~6.3 GB, 14 shards) to {parquet_dir} ...")
    snapshot_download(
        HF_REPO, repo_type="dataset", allow_patterns=["data/*.parquet"],
        local_dir=parquet_dir, max_workers=8,
    )
    shards = sorted(glob.glob(os.path.join(parquet_dir, "data", "*.parquet")))
    assert len(shards) == 14, f"expected 14 parquet shards, found {len(shards)}"

    val_dir = os.path.join(root, "val")
    for wnid in wnids:
        os.makedirs(os.path.join(val_dir, wnid), exist_ok=True)

    print("Extracting 50,000 images ...")
    per_class = Counter()
    n = 0
    for shard in shards:
        for row in pq.read_table(shard).to_pylist():
            wnid = wnids[row["label"]]
            per_class[wnid] += 1
            name = row["image"].get("path") or f"{wnid}_val_{per_class[wnid]:04d}.JPEG"
            with open(os.path.join(val_dir, wnid, os.path.basename(name)), "wb") as f:
                f.write(row["image"]["bytes"])
            n += 1
        print(f"  {os.path.basename(shard)} done ({n} images)", flush=True)

    counts = set(per_class.values())
    if not (n == NUM_IMAGES and len(per_class) == NUM_CLASSES and counts == {50}):
        sys.exit(
            f"Verification FAILED: {n} images, {len(per_class)} classes, "
            f"per-class counts {counts}. Delete {root} and retry."
        )
    shutil.rmtree(parquet_dir)
    print(f"Dataset prepared and verified at {root} ({n} images, {NUM_CLASSES} classes).")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, help="target ImageNet root directory")
    parser.add_argument(
        "--check-only", action="store_true",
        help="only report whether a prepared dataset exists at --root (exit 1 if not)",
    )
    args = parser.parse_args()

    root = os.path.abspath(args.root)
    if is_prepared(root):
        print(f"ImageNet validation set already prepared at {root} — skipping download.")
    elif args.check_only:
        sys.exit(
            f"No prepared ImageNet validation set found at {root} "
            "(expected <root>/[val/]<wnid>/*.JPEG with 1000 classes x 50 images)."
        )
    else:
        os.makedirs(root, exist_ok=True)
        download_and_extract(root)


if __name__ == "__main__":
    main()
