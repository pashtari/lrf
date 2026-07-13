#!/usr/bin/env bash
# Ablation of the QMF patch size on Kodak.
# Writes ablation_patchsize_results.json into this folder; draw the figure
# with: python ablation_patchsize/plot.py (from the repository root).
set -euo pipefail
cd "$(dirname "$0")/.."

python ablation_patchsize/eval.py --data=kodak
