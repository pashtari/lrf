"""Generate Figure 5: top-1 and top-5 ImageNet accuracy vs. bit rate.

By default the figure is drawn from your own operating points in
results/results.csv (written by src/evaluate.py). If that file is missing or
empty — or with --use-reference — the published operating points in
results/results_old.csv are used instead, reproducing the figure
exactly as in the paper without any GPU time.

Example:
    python src/plot.py
    python src/plot.py --use-reference
"""

import argparse
import os

import numpy as np
import pandas as pd

import qmf

PACKAGE_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# The figure shows the low bit rate regime.
MAX_BPP = 0.6
BPP_GRID = np.linspace(0.05, 0.5, 19)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results", default=os.path.join(PACKAGE_ROOT, "results", "results.csv")
    )
    parser.add_argument(
        "--reference", default=os.path.join(PACKAGE_ROOT, "results", "results_old.csv")
    )
    parser.add_argument(
        "--use-reference", action="store_true", help="plot the published operating points"
    )
    parser.add_argument("--output-dir", default=os.path.join(PACKAGE_ROOT, "figures"))
    return parser.parse_args()


def load_results(args) -> pd.DataFrame:
    if not args.use_reference and os.path.isfile(args.results):
        try:
            results = pd.read_csv(args.results)
        except pd.errors.EmptyDataError:
            results = pd.DataFrame()
        if not results.empty:
            print(f"Using {len(results)} operating points from {args.results}")
            return results
    print(f"Using published operating points from {args.reference}")
    return pd.read_csv(args.reference)


def main():
    args = parse_args()
    results = load_results(args)

    # To percentages, and keep the low bit rate regime shown in the figure
    results["top1_accuracy"] *= 100
    results["top5_accuracy"] *= 100
    results = results.dropna().query(f"bpp < {MAX_BPP}")
    results = results.rename(
        columns={
            "bpp": "bit rate (bpp)",
            "top1_accuracy": "top-1 accuracy (%)",
            "top5_accuracy": "top-5 accuracy (%)",
        }
    )

    # Curves are LOESS-interpolated, which needs >= 3 distinct points per
    # method with at least one inside the plotted range; drop methods that
    # are not there yet (possible mid-sweep).
    def drawable(group):
        bpp = group["bit rate (bpp)"]
        return bpp.nunique() >= 3 and bpp.between(BPP_GRID[0], BPP_GRID[-1]).any()

    kept = results.groupby("method").filter(drawable)
    dropped = set(results["method"]) - set(kept["method"])
    if dropped:
        print(f"Too few operating points yet for {sorted(dropped)} — not drawn.")
    if kept.empty:
        raise SystemExit(
            "Not enough operating points to draw any method — run more sweep "
            "points first, or use --use-reference for the published figure."
        )
    results = kept

    # Paper's legend order, restricted to the methods actually present
    methods = [m for m in ("JPEG", "SVD", "QMF") if m in set(results["method"])]

    panels = [("top-1 accuracy (%)", (5, 80)), ("top-5 accuracy (%)", (5, 95))]
    for y, ylim in panels:
        plot = qmf.Plot(results.copy())
        # LOESS interpolation over a common bpp grid, as in the paper
        plot.interpolate(x="bit rate (bpp)", y=y, x_values=BPP_GRID, groupby="method")
        plot.plot(
            x="bit rate (bpp)",
            y=y,
            groupby="method",
            errorbar="se",
            dashed=True,
            xlim=(BPP_GRID[0], BPP_GRID[-1]),
            ylim=ylim,
            legend_labels=methods,
        )
        plot.save(save_dir=args.output_dir, prefix="imagenet", format="pdf")
        print(f"Saved {y} panel to {args.output_dir}/")


if __name__ == "__main__":
    main()
