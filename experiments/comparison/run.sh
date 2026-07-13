#!/usr/bin/env bash
# Rate-distortion comparison of QMF vs JPEG and SVD on the Kodak and
# CLIC 2024 datasets. Writes {kodak,clic2024}_results.json into this folder;
# draw the figures with: python comparison/plot.py (from the repository root).
set -euo pipefail
cd "$(dirname "$0")/.."

python comparison/eval.py --data=kodak
python comparison/eval.py --data=clic2024
