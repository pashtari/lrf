import math

import numpy as np
import matplotlib.pyplot as plt
from skimage import data, img_as_float
from skimage.transform import resize
import torch
import torch.nn.functional as F
from einops import rearrange


from src.models import compression as com


def visualize_batch(images: torch.Tensor, multi_channels=True, title=None):
    """
    Visualize a batch of 2D images.

    Args:
    - images: A tensor of shape (*batch_dims, [channel], height, width)
    """
    images = com.minmax_normalize(images)

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
        ax.imshow(images[idx], cmap="gray")
        ax.axis("off")

    # Turn off any remaining unused subplots
    for ax in axs[total_subplots:]:
        ax.axis("off")

    plt.title(title, fontsize=24)
    plt.show()

    return ax


# Load an image
image = data.astronaut()
image = img_as_float(image)
image = resize(image, (224, 224))

# Visualize the image
plt.imshow(image)
plt.axis("off")
plt.title("Astronaut Image")
plt.show()

# Convert to channel-first pytorch tensor
image = torch.tensor(image).permute(-1, 0, 1).unsqueeze(0)


# Patch SVD components
svd = com.SVD()
u, v = svd.compress(image, rank=32)
print(f"Real compression ratio: {svd.real_compression_ratio:.2f}")
loss = svd.loss(image, (u, v)).item()
print(f"Relative error: {loss:.4f}")

visualize_batch(torch.einsum("bcir, bcjr -> brcij", u, v), title="SVD Rank-1 Components")


# Patch SVD components
patch_svd = com.PatchSVD(patch_size=(8, 8))
u, v = patch_svd.compress(image, rank=32)
print(f"Real compression ratio: {patch_svd.real_compression_ratio:.2f}")
loss = patch_svd.loss(image, (u, v), image.shape[-2:]).item()
print(f"Relative error: {loss:.4f}")


reshaped_u, reshaped_v = patch_svd.depatchify_uv(image, u, v)
reshaped_u = reshaped_u.transpose(1, 2)
reshaped_v = reshaped_v.transpose(1, 2)
visualize_batch(reshaped_u, title="Patch SVD Reshaped U Matrix")
visualize_batch(reshaped_v, title="Patch SVD Reshaped V Matrix")

components = torch.einsum("brchw, brdpq -> brdhpwq", reshaped_u, reshaped_v)
reshaped_components = rearrange(components, "b r c h p w q -> b r c (h p) (w q)")
visualize_batch(reshaped_components, title="Patch SVD Rank-1 Components")


# Patch LSD components
patch_lsd = com.PatchLSD(patch_size=(8, 8), num_iters=100, verbose=False)
u, v, s = patch_lsd.compress(image, rank=32, alpha=0.01)
print(f"Real compression ratio: {patch_lsd.real_compression_ratio:.2f}")
loss = patch_lsd.loss(image, (u, v, s), image.shape[-2:]).item()
print(f"Relative error: {loss:.4f}")

reshaped_u, reshaped_v = patch_lsd.depatchify_uv(image, u, v)
reshaped_u = reshaped_u.transpose(1, 2)
reshaped_v = reshaped_v.transpose(1, 2)
reshaped_s = patch_lsd.depatchify(s, image.shape[-2:])
visualize_batch(reshaped_u, title="Patch LSD Reshaped U Matrix")
visualize_batch(reshaped_v, title="Patch LSD Reshaped V Matrix")
visualize_batch(reshaped_s, title="Patch LSD Reshaped X Matrix")
