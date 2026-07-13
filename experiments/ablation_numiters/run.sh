#!/usr/bin/env bash
# Ablation of the number of QMF block-coordinate-descent iterations on Kodak.
# Writes ablation_numiters_results.json into this folder; draw the figure
# with: python ablation_numiters/plot.py (from the repository root).
set -euo pipefail
cd "$(dirname "$0")/.."

python ablation_numiters/eval.py --data=kodak
