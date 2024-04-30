import numpy as np
import torch
from torchvision.transforms import v2
import matplotlib.pyplot as plt
from skimage.io import imread
from einops import rearrange

import lrf


def visualize_batch(images: torch.Tensor, multi_channels=True, title=None, **kwargs):
    """
    Visualize a batch of 2D images.

    Args:
    - images: A tensor of shape (*batch_dims, [channel], height, width)
    """
    images = lrf.minmax_normalize(images)

    # Ensure the tensor is on the CPU
    images = images.cpu().numpy()

    if images.ndim == 2:
        images = images[None, ...]

    if multi_channels:
        images = np.einsum("... c m n -> ... m n c", images)
        batch_dims = images.shape[:-3]
    else:
        batch_dims = images.shape[:-2]

    # Compute the number of subplots needed
    total_subplots = np.prod(batch_dims)

    # Compute the grid size for subplots
    grid_size = int(np.ceil(np.sqrt(total_subplots)))

    # Create the subplots
    fig, axs = plt.subplots(grid_size, grid_size, figsize=(15, 15))

    # Flatten the axes for easy indexing
    axs = [axs] if grid_size == 1 else axs.ravel()

    # Iterate over the batch dimensions and visualize each image
    for idx in np.ndindex(batch_dims):
        ax = axs[
            idx[0] if len(batch_dims) == 1 else np.ravel_multi_index(idx, batch_dims)
        ]
        ax.imshow(images[idx], **kwargs)
        ax.axis("off")

    # Turn off any remaining unused subplots
    for ax in axs[total_subplots:]:
        ax.axis("off")

    plt.title(title, fontsize=24)
    plt.show()

    return ax


# Load the image
image = imread("./data/kodak/kodim23.png")

# Transform the input image
transforms = v2.Compose([v2.ToImage()])
image = torch.tensor(transforms(image))

# Visualize image
plt.imshow(image.permute(1, 2, 0))
plt.axis("off")
plt.title("Original Image")
plt.show()


# Convert to channel-first pytorch tensor
enocoded_image = lrf.imf_encode(
    image,
    color_space="YCbCr",
    scale_factor=(0.5, 0.5),
    quality=(10, 5, 5),
    patch=True,
    patch_size=(8, 8),
    bounds=(-16, 15),
    dtype=torch.int8,
    num_iters=10,
    verbose=False,
)
real_bpp = lrf.get_bbp(image.shape[-2:], enocoded_image)
print(f"bpp: {real_bpp}")

encoded_metadata, encoded_factors = lrf.separate_bytes(enocoded_image, 2)
metadata = lrf.bytes_to_dict(encoded_metadata)
encoded_factors = lrf.separate_bytes(encoded_factors, 6)

R1, R2, R3 = metadata["rank"]
u_y, v_y = lrf.batched_pil_decode(encoded_factors[0], R1), lrf.batched_pil_decode(
    encoded_factors[1], R1
)
u_cb, v_cb = lrf.batched_pil_decode(encoded_factors[2], R2), lrf.batched_pil_decode(
    encoded_factors[3], R2
)
u_cr, v_cr = lrf.batched_pil_decode(encoded_factors[4], R3), lrf.batched_pil_decode(
    encoded_factors[5], R3
)

factors = []
for i, (u, v) in enumerate(((u_y, v_y), (u_cb, v_cb), (u_cr, v_cr))):
    bounds = metadata["bounds"]

    u, v = u[:, 0:1, ...].to(torch.uint8), v[:, 0:1, ...].to(torch.uint8)

    factors.append((u, v))


visualize_batch(factors[0][0], title="IMF Components (U) - Y", cmap="gray")
visualize_batch(factors[1][0], title="IMF Components (U) - Cb", cmap="coolwarm")
visualize_batch(factors[2][0], title="IMF Components (U) - Cr", cmap="coolwarm")

visualize_batch(factors[0][1], title="IMF Coefficients (V) - Y", cmap="gray")
visualize_batch(factors[1][1], title="IMF Coefficients (V) - Cb", cmap="coolwarm")
visualize_batch(factors[2][1], title="IMF Coefficients (V) - Cr", cmap="coolwarm")


terms = []
for i, (u, v) in enumerate(((u_y, v_y), (u_cb, v_cb), (u_cr, v_cr))):
    bounds = metadata["bounds"]
    u, v = u[:, 0:1, ...].float() + bounds[0], v[:, 0:1, ...].float() + bounds[0]

    x = torch.einsum("r c h w, r c p q -> r c h p w q", u, v)
    x = rearrange(x, "r c h p w q -> r c (h p) (w q)")

    terms.append(x)

visualize_batch(terms[0], title="IMF Rank-1 Terms - Y", cmap="gray")
visualize_batch(terms[1], title="IMF Rank-1 Terms - Cb", cmap="coolwarm")
visualize_batch(terms[2], title="IMF Rank-1 Terms - Cr", cmap="coolwarm")
