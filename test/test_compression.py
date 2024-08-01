import torch
from skimage import data
from matplotlib import pyplot as plt

import lrf

# Load the astronaut image from skimage and convert it to YCbCr
img_rgb = torch.from_numpy(data.astronaut()).permute(2, 0, 1)

img_ycbcr = lrf.rgb_to_ycbcr(img_rgb)
img_rgb_hat = lrf.ycbcr_to_rgb(img_ycbcr)
print(torch.square(img_rgb - img_rgb_hat).mean())

# Perform chroma subsampling with a 4:2:0 ratio
downsampled_ycbcr = lrf.chroma_downsampling(
    img_ycbcr, scale_factor=(0.5, 0.5), mode="area"
)

# Perform chroma upsampling with a 4:2:0 ratio
img_ycbcr_hat = lrf.chroma_upsampling(
    downsampled_ycbcr, scale_factor=(2, 2), mode="nearest"
)
img_rgb_hat = lrf.ycbcr_to_rgb(img_ycbcr_hat).clip(0, 255)

# Plot the original and downsampled images
fig, ax = plt.subplots(2, 4, figsize=(40, 20))
ax[0, 0].imshow(img_rgb.permute(1, 2, 0))
ax[0, 0].set_title("Original Image (RGB)")
ax[0, 0].axis("off")

ax[0, 1].imshow(img_ycbcr[0], cmap="gray")
ax[0, 1].set_title("Original Image (Y)")
ax[0, 1].axis("off")

ax[0, 2].imshow(img_ycbcr[1], cmap="gray")
ax[0, 2].set_title("Original Image (Cb)")
ax[0, 2].axis("off")

ax[0, 3].imshow(img_ycbcr[2], cmap="gray")
ax[0, 3].set_title("Original Image (Cr)")
ax[0, 3].axis("off")

ax[1, 0].imshow(img_rgb_hat.to(torch.uint8).permute(1, 2, 0))
ax[1, 0].set_title("Downsampled Image (RGB)")
ax[1, 0].axis("off")

ax[1, 1].imshow(downsampled_ycbcr[0][0], cmap="gray")
ax[1, 1].set_title("Downsampled Image (Y)")
ax[1, 1].axis("off")

ax[1, 2].imshow(downsampled_ycbcr[1][0], cmap="gray")
ax[1, 2].set_title("Downsampled Image (Cb)")
ax[1, 2].axis("off")

ax[1, 3].imshow(downsampled_ycbcr[2][0], cmap="gray")
ax[1, 3].set_title("Downsampled Image (Cr)")
ax[1, 3].axis("off")

plt.show()
