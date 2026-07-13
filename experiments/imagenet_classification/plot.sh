#!/usr/bin/env bash
# Generate the two panels of Figure 5 (top-1 / top-5 accuracy vs. bit rate)
# into ./figures, from your own results/results.csv if it exists, otherwise
# from the published operating points. Extra arguments are forwarded, e.g.:
#     bash plot.sh --use-reference
set -euo pipefail
cd "$(dirname "$0")"

python src/plot.py "$@"
