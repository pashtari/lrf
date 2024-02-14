import math

import numpy as np
import matplotlib.pyplot as plt
from skimage import data, img_as_float
from skimage.transform import resize
import torch
import torch.nn.functional as F
from einops import rearrange


def min_max(x):
    x -= x.min(-1, keepdim=True)[0].min(-2, keepdim=True)[0]
    x /= x.max(-1, keepdim=True)[0].max(-2, keepdim=True)[0]
    return x


def visualize_batch(images: torch.Tensor, multi_channels=True, title=None):
    """
    Visualize a batch of 2D images.

    Args:
    - images: A tensor of shape (*batch_dims, [channel], height, width)
    """
    images = min_max(images)

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


# def get_svd_components(x, compression_ratio=None):
#     # Convert the image to a PyTorch tensor
#     x = torch.tensor(x, dtype=torch.float32).permute(-1, 0, 1).unsqueeze(0)

#     H, W = x.shape[-2:]
#     R = math.floor(H * W // (compression_ratio * (H + W)))

#     u, s, vt = torch.linalg.svd(x, full_matrices=False)
#     components = torch.einsum(
#         "b c h r, b c r, b c r w -> b r c h w", u[..., :, :R], s[..., :R], vt[..., :R, :]
#     )
#     return components


def get_svd_patches_components(x, compression_ratio=None, patch_size=8):
    # Convert the image to a PyTorch tensor
    x = torch.tensor(x, dtype=torch.float32).permute(-1, 0, 1).unsqueeze(0)

    # Rearrange image to form columns of flattened patches
    patches = rearrange(
        x, "b c (h p1) (w p2) -> b (h w) (c p1 p2)", p1=patch_size, p2=patch_size
    )

    M, N = patches.shape[-2:]
    R = math.floor(M * N // (compression_ratio * (M + N)))

    u, s, vt = torch.linalg.svd(patches, full_matrices=False)
    components = torch.einsum(
        "b m r, b r, b r n -> b r m n", u[:, :, :R], s[:, :R], vt[:, :R, :]
    )

    components = rearrange(
        components,
        "b r (h w) (c p1 p2) -> b r c (h p1) (w p2)",
        p1=patch_size,
        p2=patch_size,
        h=x.shape[-2] // patch_size,
    )
    return components


def get_svd_patches_u(x, compression_ratio=1, patch_size=8):
    # Convert the image to a PyTorch tensor
    x = torch.tensor(x, dtype=torch.float32).permute(-1, 0, 1).unsqueeze(0)

    # Rearrange image to form columns of flattened patches
    patches = rearrange(
        x, "b c (h p1) (w p2) -> b (h w) (c p1 p2)", p1=patch_size, p2=patch_size
    )

    M, N = patches.shape[-2:]
    R = math.floor(M * N // (compression_ratio * (M + N)))

    u, s, vt = torch.linalg.svd(patches, full_matrices=False)
    u_reshaped = rearrange(
        u[:, :, :R],
        "b (h w) r -> b r () h w",
        h=x.shape[-2] // patch_size,
    )
    return u_reshaped


def get_svd_patches_v(x, compression_ratio=1, patch_size=8):
    # Convert the image to a PyTorch tensor
    x = torch.tensor(x, dtype=torch.float32).permute(-1, 0, 1).unsqueeze(0)

    # Rearrange image to form columns of flattened patches
    patches = rearrange(
        x, "b c (h p1) (w p2) -> b (h w) (c p1 p2)", p1=patch_size, p2=patch_size
    )

    M, N = patches.shape[-2:]
    R = math.floor(M * N // (compression_ratio * (M + N)))

    u, s, vt = torch.linalg.svd(patches, full_matrices=False)
    v_reshaped = rearrange(
        vt[:, :R, :],
        "b r (c p1 p2) -> b r c p1 p2",
        p1=patch_size,
        p2=patch_size,
    )
    return v_reshaped


def get_patch_svd_u_reshaped(x, patch_size=8):
    # Convert the image to a PyTorch tensor
    x = torch.tensor(x, dtype=torch.float32).permute(-1, 0, 1).unsqueeze(0)

    # Rearrange image to form columns of flattened patches
    patches = rearrange(
        x, "b c (h p1) (w p2) -> b (h w) (c p1 p2)", p1=patch_size, p2=patch_size
    )
    u, s, vt = torch.linalg.svd(patches, full_matrices=False)
    u = u[:, :, :48]
    # u_reshaped = rearrange(u, "b (h w) (c p1 p2) -> b c (h p1) (w p2)", p1=4, p2=4, h=28)

    x_low_reshaped = F.interpolate(x, size=(112, 112), mode="bilinear")
    x_low = rearrange(
        x_low_reshaped, "b c (h p1) (w p2) -> b (h w) (c p1 p2)", p1=4, p2=4
    )
    a = torch.linalg.lstsq(
        torch.cat((u, 100 * torch.eye(48).unsqueeze(0)), 1),
        torch.cat((x_low, torch.zeros((1, 48, 48))), 1),
    )[0]
    u_low = u @ a
    u_low_shaped = rearrange(
        u_low, "b (h w) (c p1 p2) -> b c (h p1) (w p2)", p1=4, p2=4, h=28
    )
    return u_low_shaped, x_low_reshaped


# Load an image
image = data.astronaut()
image = img_as_float(image)
image = resize(image, (224, 224))

# Visualize the image
plt.imshow(image)
plt.axis("off")
plt.title("Astronaut Image")
plt.show()

compression_ratio = 5

# # SVD components
# svd_components = get_svd_components(image, compression_ratio=compression_ratio)
# visualize_batch(svd_components, title="SVD Components")

# SVD patches components
svd_patches_components = get_svd_patches_components(
    image, compression_ratio=compression_ratio
)
visualize_batch(svd_patches_components, title="Patch SVD components")

svd_patches_u = get_svd_patches_u(image, compression_ratio=compression_ratio)
visualize_batch(svd_patches_u, title="Patch SVD U Matrix")

svd_patches_v = get_svd_patches_v(image, compression_ratio=compression_ratio)
visualize_batch(svd_patches_v, title="Patch SVD V Matrix")

svd_patches_u_reshaped, svd_patches_x_reshaped = get_patch_svd_u_reshaped(image)
visualize_batch(svd_patches_u_reshaped, title="Patch SVD Reshaped U Matrix")
visualize_batch(svd_patches_x_reshaped, title="Patch SVD Reshaped X Matrix")


fig, axs = plt.subplots(1, 2)
axs[0].imshow(min_max(svd_patches_u_reshaped).squeeze(0).permute(1, 2, 0))
axs[1].imshow(min_max(svd_patches_x_reshaped).squeeze(0).permute(1, 2, 0))
plt.show()

fig, axs = plt.subplots(1, 2)
axs[0].imshow(svd_patches_u_reshaped.squeeze(0).permute(1, 2, 0))
axs[1].imshow(svd_patches_x_reshaped.squeeze(0).permute(1, 2, 0))
plt.show()
