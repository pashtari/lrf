This repo provides PyTorch implementation of low-rank factorization (LRF) methods for data compression. It also includes the official implementation of "Quantization-free Lossy Image Compression Using Integer Matrix Factorization".

# Introduction
Lossy image compression is essential for efficient transmission and storage. Traditional compression methods mainly rely on discrete cosine transform (DCT) or singular value decomposition (SVD), both of which represent image data in continuous domains and therefore necessitate carefully designed quantizers. Notably, SVD-based methods are more sensitive to quantization errors than DCT-based methods like JPEG. To address this issue, we introduce a variant of integer matrix factorization (IMF) to develop a novel quantization-free lossy image compression method. IMF provides a low-rank representation of the image data as a product of two smaller factor matrices with bounded integer elements, thereby eliminating the need for quantization. We propose an efficient, provably convergent iterative algorithm for IMF using a block coordinate descent (BCD) scheme, with subproblems having closed-form solutions. Our experiments on the Kodak and CLIC 2024 datasets demonstrate that our IMF compression method consistently outperforms JPEG at low bit rates below 0.25 bits per pixel (bpp) and remains comparable at higher bit rates. We also assessed our method's capability to preserve visual semantics by evaluating an ImageNet pre-trained classifier on compressed images. Remarkably, our method improved top-1 accuracy by over 5 percentage points compared to JPEG at bit rates under 0.25 bpp.


# Install Requirements
First make sure that you have already installed [PyTorch](https://pytorch.org/get-started/locally/) since the details on how to do this depend on whether you have a CPU, GPU, etc.
Then, clone the repository and install the requirements:
```bash
$ pip install git+https://github.com/pashtari/lrf.git
$ cd root_dir/experiments/examples
$ pip install -r requirements.txt
```

# Get Started
**Import packages**
```python
import os

import numpy as np
import torch
import seaborn as sns

import lrf
from lrf.utils import utils
```
**Set output directory**
```python
script_dir = os.path.join(os.path.abspath(""), "experiments/examples")
save_dir = os.path.join(script_dir, "fig4_first_row")
prefix = "fig4_first_row"
```
**Read the image**
```python
image = utils.read_image("./data/kodak/kodim01.png")
```
**Visualize the image**
```python
utils.vis_image(image, save_dir=save_dir, prefix=prefix, format="pdf")
```
![image](https://github.com/user-attachments/assets/01df1153-f9ae-4b5d-bc9c-bc5852425ff6)

**Calculate compression metrics**
```python
results = []

# JPEG
for quality in range(0, 25, 1):
    params = {"quality": quality}
    config = {"data": prefix, "method": "JPEG", **params}
    log = utils.eval_compression(
        image,
        lrf.pil_encode,
        lrf.pil_decode,
        reconstruct=True,
        format="JPEG",
        **params,
    )
    results.append({**config, **log})

# SVD
for quality in np.linspace(0.0, 4, 50):
    params = {
        "color_space": "RGB",
        "quality": quality,
        "patch": True,
        "patch_size": (8, 8),
    }
    config = {"data": prefix, "method": "SVD", **params}
    log = utils.eval_compression(
        image, lrf.svd_encode, lrf.svd_decode, reconstruct=True, **params
    )
    results.append({**config, **log})


# IMF
for quality in np.linspace(0, 25, 75):
    params = {
        "color_space": "YCbCr",
        "scale_factor": (0.5, 0.5),
        "quality": (quality, quality / 2, quality / 2),
        "patch": True,
        "patch_size": (8, 8),
        "bounds": (-16, 15),
        "dtype": torch.int8,
        "num_iters": 10,
        "verbose": False,
    }
    config = {"data": prefix, "method": "IMF", **params}
    log = utils.eval_compression(
        image, lrf.imf_encode, lrf.imf_decode, reconstruct=True, **params
    )
    results.append({**config, **log})
```

**Make qualitative comparisons**
```python
bpps = [0.2]
utils.vis_collage(results, bpps=bpps, save_dir=save_dir, prefix=prefix, format="pdf")
```
![image](https://github.com/user-attachments/assets/a4c66890-2eb2-4cf2-a02c-b467ea221850)


# Contact
This repo is currently maintained by Pooya Ashtari ([@pashtari](https://github.com/pashtari)) and Pourya Behmandpoor ([@pourya-b](https://github.com/pourya-b)).