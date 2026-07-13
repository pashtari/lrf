#!/usr/bin/env bash
# Ablation of the QMF quantization bounds (alpha, beta) on Kodak.
# Writes ablation_bounds_results.json into this folder; draw the figure
# with: python ablation_bounds/plot.py (from the repository root).
set -euo pipefail
cd "$(dirname "$0")/.."

python ablation_bounds/eval.py --data=kodak
