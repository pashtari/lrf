#!/usr/bin/env bash
# Ablation of the QMF color space (RGB vs YCbCr) on Kodak.
# Writes ablation_colorspace_results.json into this folder; draw the figure
# with: python ablation_colorspace/plot.py (from the repository root).
set -euo pipefail
cd "$(dirname "$0")/.."

python ablation_colorspace/eval.py --data=kodak
