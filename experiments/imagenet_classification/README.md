# ImageNet Classification under Compression (Figure 5)

Reproduces Figure 5 of [*"Quantization-aware Matrix Factorization for Low Bit Rate Image Compression"*](https://doi.org/10.1016/j.ins.2025.122646) (Information Sciences, 2025): top-1 and top-5 accuracy of a pretrained ResNet-50 on the **ImageNet-1k validation set** (50,000 images) after compressing the images with **QMF**, **JPEG**, or **SVD** at low bit rates.

**The pipeline** (one operating point = one method at one quality):

```
full-resolution RGB uint8 image
  → compress + decompress (QMF / JPEG / SVD)   ← bpp measured HERE, at original resolution
  → Resize(256, bilinear, antialias=True)
  → CenterCrop(224)
  → scale to [0,1] float32
  → Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
  → ResNet-50 (torchvision, ResNet50_Weights.IMAGENET1K_V2, no fine-tuning)
  → top-1 / top-5 accuracy + mean bpp over the 50k images
```

```
imagenet_classification/
├── run.sh                       ← run everything: dataset (first use) + all sweeps
├── plot.sh                      ← draw the figure from results/results.csv
├── src/
│   ├── evaluate.py              ← one operating point → one CSV row
│   ├── download_imagenet_val.py ← one-time dataset download (idempotent)
│   ├── plot.py                  ← draw both panels of Figure 5
│   └── imagenet_wnids.txt       ← the 1000 wnids in canonical label order
├── results/
│   └── results.csv              ← one row per operating point
│                                  (method, quality, bpp, top1_accuracy, top5_accuracy)
└── figures/                     ← output: imagenet_top-{1,5}_accuracy.pdf
```

The committed `results/results.csv` holds the full 80-point sweep, so you can inspect the numbers or draw the figure immediately; your own runs replace these rows point by point (`--force`) or extend them.

## 1. Setup

Linux, Python ≥ 3.10, and a CUDA GPU with ≥ 8 GB memory (used automatically; CPU works but is slow). Compression runs on the CPU inside the dataloader, so more cores help (`--num-workers`).

```bash
pip install torch torchvision              # pick your CUDA variant from pytorch.org
pip install git+https://github.com/pashtari/qmf.git
pip install -r requirements.txt
```

Use the **`qmf`** package, not the older pre-publication `lrf` package.

## 2. Dataset

The ILSVRC2012 validation set in the extracted layout `<root>/[val/]<wnid>/*.JPEG`. Either do nothing — `run.sh` downloads it once from a public mirror (`mrm8488/ImageNet1K-val`, ~6.3 GB, no account needed) into the repository's `data/imagenet` and verifies it (50,000 images, 1000 classes × 50) — or point the scripts at an existing copy with `IMAGENET_DIR=<dir>`. ImageNet has its own [terms of access](https://image-net.org/download.php); download only if you agree to them.

Labels follow the canonical ILSVRC2012/torchvision convention (index of the wnid in the sorted list of all 1000 wnids), the same convention the pretrained weights use — no remapping.

## 3. Operating points

**QMF** — `qmf.qmf_encode` with only `quality` set; everything else is the package default:

| Parameter | Value |
|---|---|
| color space | `YCbCr`, chroma downsampled ×2 |
| patching | `patch=True`, `patch_size=(8, 8)` |
| bounds (α, β) | `(-16, 15)` |
| factor dtype | `int8` |
| iterations | `num_iters=10` |
| rank | from `quality`: `R = max(round(min(M, N) · q/100), 1)` for the patchified matrix; chroma channels use `q/2` |

**JPEG** — Pillow codec. **SVD** — package defaults.

The published grids were QMF 0–29.5 (step 0.5), JPEG 0–75 (step 1), SVD 0–6.0 (step 0.2); only points below 0.8 bpp enter the figure, whose curves cover 0.05–0.5 bpp. `run.sh` therefore sweeps just the figure-relevant portions (QMF 0–15.5, JPEG 0–26, SVD 0–4.0 — **80 points**); extend the `sweep` lines for the full grids.

## 4. Run

```bash
bash run.sh                                  # dataset auto-downloaded on first use
IMAGENET_DIR=/path/to/imagenet bash run.sh   # or use an existing copy
```

~3–4 hours on a modern single-GPU desktop (~2–5 min per point). Each finished point appends one row to `results/results.csv`; points already in the CSV are skipped, so you can interrupt and re-launch the same command at any time. Per-point output:

```
[#####                         ]   6/32  qmf quality=2.5
ResNet-50 (ResNet50_Weights.IMAGENET1K_V2) on cuda; 50000 images
qmf quality=2.5: 100%|██████████| 196/196 [02:41<00:00]
QMF quality=2.5 -> bpp 0.12377, top-1 0.23312, top-5 0.41958 (saved to .../results/results.csv)
```

A single operating point (also how to reproduce any individual reported accuracy):

```bash
python src/evaluate.py --method qmf --quality 5.0
# QMF quality=5 -> bpp 0.19307, top-1 0.50216, top-5 0.74078
```

Options: `--imagenet-dir`, `--results`, `--batch-size` (256), `--num-workers` (8), `--device`, `--force` (re-run a point, replacing its row). `--method none` gives the uncompressed baseline (top-1 0.80858). The pipeline is deterministic: rerunning a point reproduces its numbers exactly on the same software stack.

**Quick sanity check** — the dataset class accepts any class *subset* in the same layout (`subset/<wnid>/*.JPEG`) with correct canonical labels, so you can validate your setup on a few labeled validation images in minutes: at quality 25 expect top-1 ≈ 0.9 on typical images, at quality 5 roughly half.

```bash
python src/evaluate.py --method qmf --quality 25 --imagenet-dir /path/to/subset --results results/sanity.csv
```

## 5. Plot

```bash
bash plot.sh                  # draws the figure from results/results.csv
```

Writes the two panels of Figure 5 to `figures/imagenet_top-{1,5}_accuracy.pdf`. As in the paper, the operating points are LOESS-interpolated onto a common bpp grid; dashed segments mark extrapolation. Methods with fewer than 3 points so far are omitted until the sweep produces more.

## What to expect when you rerun

This package was validated with a full re-run on the 50k validation set with current package versions (July 2026: torch 2.10, torchvision 0.25, current `qmf`). JPEG and SVD reproduce the published numbers essentially exactly (mean |Δ| ≤ 0.002). QMF accuracies at a given `quality` also match, but its encoder has improved since the 2024 runs — the same quality now yields ~5–15 % smaller files at mid/high quality — so the accuracy-vs-bitrate curve sits **above** the published one: by ≈ 3–5 top-1 points around 0.15–0.2 bpp and ≈ 1–2 points elsewhere. A faithful rerun matches or exceeds the published QMF curve; if yours is *worse*, check the pitfalls below.

## Common pitfalls

1. **Weights.** Must be `ResNet50_Weights.IMAGENET1K_V2` (not V1; `pretrained=True` on older torchvision means V1, costing ≈ 4–5 top-1 points here).
2. **Order and bpp.** Compress the **full-resolution** image first, then Resize/CenterCrop; measure bpp at the **original** resolution, not on the 224×224 crop.
3. **QMF parameters.** Use `quality` (not `rank`) with the defaults in section 3 — especially `bounds=(-16, 15)`, `num_iters=10`, `patch_size=(8, 8)`, YCbCr.
4. **Package.** The current `qmf`, not the pre-publication `lrf`.
5. **Preprocessing.** Bilinear with `antialias=True`, RGB, CHW, normalization after scaling to [0, 1].

To localize a discrepancy, diff your numbers against the committed `results.csv` at matching `(method, quality)`: a bpp shift points to the bpp measurement, an accuracy shift to the model/preprocessing, low-quality-only divergence to the compression parameters.

## Contact

Pooya Ashtari — pooya.ash@gmail.com
