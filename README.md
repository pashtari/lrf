![LRF Logo](figures/logo.svg)

This repository provides a PyTorch implementation of **low-rank factorization (LRF) methods for data compression**. Particularly, it includes the official implementation of [*"Quantization-aware Matrix Factorization for Low Bit Rate Image Compression."*](https://arxiv.org/abs/2408.12691)


<table style="border-collapse: collapse; table-layout: fixed; width: 100%;">
  <tr>
    <td style="text-align: center; border: none; width: 45%;">
      <figure style="margin: 0; padding: 0;">
        <img src="figures/clic_flower.png" alt="Original" width="100%">
        <figcaption>Original<br>&nbsp</figcaption>
      </figure>
    </td>
    <td style="text-align: center; border: none; width: 45%;">
      <figure style="margin: 0; padding: 0;">
        <img src="figures/clic_flower_jpeg_bpp_0.14_psnr_22.66.png" alt="JPEG" width="100%">
        <figcaption>JPEG<br>(bitrate: 0.14 bpp, PSNR: 22.66 dB)</figcaption>
      </figure>
    </td>
  </tr>
  <tr>
    <td style="text-align: center; border: none; width: 45%;">
      <figure style="margin: 0; padding: 0;">
        <img src="figures/clic_flower_svd_bpp_0.12_psnr_26.90.png" alt="SVD" width="100%">
        <figcaption>SVD<br>(bitrate: 0.12 bpp, PSNR: 26.90 dB)</figcaption>
      </figure>
    </td>
    <td style="text-align: center; border: none; width: 45%;">
      <figure style="margin: 0; padding: 0;">
        <img src="figures/clic_flower_qmf_bpp_0.12_psnr_31.63.png" alt="QMF" width="100%">
        <figcaption><b>QMF</b><br>(bitrate: 0.12 bpp, PSNR: 31.63 dB)</figcaption>
      </figure>
    </td>
  </tr>
</table>


## Installation

First, ensure that you have [PyTorch](https://pytorch.org/get-started/locally/) installed. The installation process may vary depending on your hardware (CPU, GPU, etc.).

Next, install the `lrf` package:

```bash
$ pip install git+https://github.com/pashtari/lrf.git
```


## Quick Start

This guide will help you get started with the quantization-aware matrix factorization (QMF) compression method using the `kodim01` image from the Kodak dataset. For a more detailed example comparing QMF against JPEG and SVD, check out [this notebook](experiments/examples/comparison.ipynb). To better understand each step of the QMF compression using visualizations, refer to [this notebook](experiments/examples/qmf_pipeline.ipynb).

**Import Packages**
```python
import torch

import lrf
```

**Load Image**
```python
image = lrf.read_image("./kodim01.png")
```

**QMF Encode Image**
```python
qmf_encoded = lrf.qmf_encode(
    image,
    color_space="YCbCr",
    scale_factor=(0.5, 0.5),
    quality=7,
    patch=True,
    patch_size=(8, 8),
    bounds=(-16, 15),
    dtype=torch.int8,
    num_iters=10,
)
```

**Decode QMF-Encoded Image**
```python
image_qmf = lrf.qmf_decode(qmf_encoded)
```

**Calculate Compression Metrics**
```python
cr_value = lrf.compression_ratio(image, qmf_encoded)
bpp_value = lrf.bits_per_pixel(image.shape[-2:], qmf_encoded)
psnr_value = lrf.psnr(image, image_qmf)
ssim_value = lrf.ssim(image, image_qmf)

metrics = {
    "compression ratio": cr_value,
    "bit rate (bpp)": bpp_value,
    "PSNR (dB)": psnr_value,
    "SSIM": ssim_value,
}
print(metrics)
```

```plaintext
{
    "compression ratio": 117.040,
    "bit rate (bpp)": 0.205,
    "PSNR (dB)": 21.928,
    "SSIM": 0.511
}
```

**Visualize Original and Compressed Images**
```python
lrf.vis_image(image, title="Original")
lrf.vis_image(
    image_qmf, title=f"QMF (bit rate = {bpp_value:.2f} bpp, PSNR = {psnr_value:.2f} dB)"
)
```

<table style="border-collapse: collapse; table-layout: fixed; width: 100%;">
  <tr>
    <td style="text-align: center; border: none; width: 45%;">
      <figure style="margin: 0; padding: 0;">
        <img src="figures/kodim01.png" alt="Original" width="100%">
        <figcaption>Original<br>&nbsp</figcaption>
      </figure>
    </td>
    <td style="text-align: center; border: none; width: 45%;">
      <figure style="margin: 0; padding: 0;">
        <img src="figures/kodim01_jpeg_bpp_0.21_psnr_20.22.png" alt="JPEG" width="100%">
        <figcaption>JPEG<br>(bitrate: 0.21 bpp, PSNR: 20.22 dB)</figcaption>
      </figure>
    </td>
  </tr>
  <tr>
    <td style="text-align: center; border: none; width: 45%;">
      <figure style="margin: 0; padding: 0;">
        <img src="figures/kodim01_svd_bpp_0.22_psnr_20.24.png" alt="SVD" width="100%">
        <figcaption>SVD<br>(bitrate: 0.22 bpp, PSNR: 20.24 dB)</figcaption>
      </figure>
    </td>
    <td style="text-align: center; border: none; width: 45%;">
      <figure style="margin: 0; padding: 0;">
        <img src="figures/kodim01_qmf_bpp_0.21_psnr_21.93.png" alt="QMF" width="100%">
        <figcaption><b>QMF</b><br>(bitrate: 0.21 bpp, PSNR: 21.93 dB)</figcaption>
      </figure>
    </td>
  </tr>
</table>


## Contact
This repo is currently maintained by Pooya Ashtari ([@pashtari](https://github.com/pashtari)) and Pourya Behmandpoor ([@pourya-b](https://github.com/pourya-b)). Feel free to reach out for any queries or contributions.