#!/usr/bin/env bash
# Generate the two panels of Figure 5 (top-1 / top-5 accuracy vs. bit rate)
# into ./figures from results/results.csv. Extra arguments are forwarded.
set -euo pipefail
cd "$(dirname "$0")"

python src/plot.py "$@"
