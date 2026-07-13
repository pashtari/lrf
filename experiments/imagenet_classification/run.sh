#!/usr/bin/env bash
# Run the ImageNet classification experiment behind Figure 5 of the paper:
# sweep QMF, JPEG and SVD over the figure-relevant portions of their quality
# grids (80 operating points, all landing below ~0.8 bpp) on the ImageNet-1k
# validation set. Results accumulate in results/results.csv; generate the
# figure afterwards with plot.sh.
#
# Usage:
#     bash run.sh                                  # auto-download dataset
#     IMAGENET_DIR=/path/to/imagenet bash run.sh   # use an existing copy
#
# If IMAGENET_DIR is not set, the validation set is downloaded once to the
# repository's data folder, ../../data/imagenet (~6.9 GB), and reused on
# subsequent runs. Expect roughly 3-4 hours in total on a modern single-GPU
# desktop. Everything is resumable: operating points already in
# results/results.csv are skipped, so the script can be interrupted and
# re-launched with the same command.

set -euo pipefail
cd "$(dirname "$0")"
export LC_ALL=C  # so seq always emits '.' decimals regardless of locale

if [ -n "${IMAGENET_DIR:-}" ]; then
    # User-supplied dataset: verify it, never download into it.
    python src/download_imagenet_val.py --root "$IMAGENET_DIR" --check-only
else
    IMAGENET_DIR=../../data/imagenet
    python src/download_imagenet_val.py --root "$IMAGENET_DIR"
fi

# progress <current> <total> <label> — one-line progress bar for the sweep
progress () {
    local cur=$1 total=$2 label=$3 width=30 filled bar
    filled=$(( cur * width / total ))
    bar="$(printf '%*s' "$filled" '' | tr ' ' '#')"
    printf '[%-*s] %3d/%-3d %s\n' "$width" "$bar" "$cur" "$total" "$label"
}

# sweep <method> <quality>... — evaluate every quality; done points skip fast
sweep () {
    local method=$1 total i=0
    shift
    total=$#
    for q in "$@"; do
        i=$((i + 1))
        progress "$i" "$total" "$method quality=$q"
        python src/evaluate.py --method "$method" --quality "$q" --imagenet-dir "$IMAGENET_DIR"
    done
}

sweep qmf  $(seq 0 0.5 15.5)
sweep jpeg $(seq 0 1 26)
sweep svd  $(seq 0 0.2 4.0)

echo "All sweeps complete -> results/results.csv. Generate the figure with: bash plot.sh"
