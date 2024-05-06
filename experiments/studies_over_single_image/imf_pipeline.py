import numpy as np
import torch
from torch.nn.modules.utils import _pair
from torchvision.transforms import v2
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from skimage.io import imread
from einops import rearrange

import lrf


def visualize_batch(
    images: torch.Tensor,
    multi_channels=True,
    figsize=None,
    grid_size=None,
    title=None,
    **kwargs,
):
    """
    Visualize a batch of 2D images.

    Args:
    - images: A tensor of shape (*batch_dims, [channel], height, width)
    """
    # images = lrf.minmax_normalize(images)

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
    grid_size = int(np.ceil(np.sqrt(total_subplots))) if grid_size is None else grid_size
    grid_size = _pair(grid_size)

    # Create the subplots
    fig, axs = plt.subplots(*grid_size, figsize=figsize)

    # Flatten the axes for easy indexing
    axs = [axs] if grid_size[0] * grid_size[1] == 1 else axs.ravel()

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

    fig.tight_layout(pad=0.75)
    plt.title(title, fontsize=24)
    plt.show()

    return fig, axs


image_name = "kodim23"

# Load the image
image = imread(f"./data/kodak/{image_name}.png")

# Transform the input image
transforms = v2.Compose([v2.ToImage()])
image = torch.tensor(transforms(image))

# Visualize image
fig, axs = visualize_batch(image.unsqueeze(0))
fig.savefig(f"./paper/figures/{image_name}.pdf", bbox_inches="tight", pad_inches=0)

# Convert to YCbCr
ycbcr = lrf.rgb_to_ycbcr(image) / 255

fig, ax = plt.subplots()
ax.imshow(ycbcr[0:1].permute(1, 2, 0), cmap="gray")
ax.axis("off")
plt.show()
fig.savefig(f"./paper/figures/{image_name}_y.pdf", bbox_inches="tight", pad_inches=0)

fig, ax = plt.subplots()
ax.imshow(ycbcr[1:2].permute(1, 2, 0), cmap="coolwarm")
ax.axis("off")
plt.show()
fig.savefig(f"./paper/figures/{image_name}_cb.pdf", bbox_inches="tight", pad_inches=0)

fig, ax = plt.subplots()
ax.imshow(ycbcr[2:3].permute(1, 2, 0), cmap="coolwarm")
ax.axis("off")
plt.show()
fig.savefig(f"./paper/figures/{image_name}_cr.pdf", bbox_inches="tight", pad_inches=0)

# Chroma downsampling
yd, cbd, crd = lrf.chroma_downsampling(ycbcr, scale_factor=(0.5, 0.5), mode="area")

fig, axs = visualize_batch(yd.unsqueeze(0), cmap="gray")
fig.savefig(f"./paper/figures/{image_name}_yd.pdf", bbox_inches="tight", pad_inches=0)

fig, axs = visualize_batch(cbd.unsqueeze(0), cmap="coolwarm")
fig.savefig(f"./paper/figures/{image_name}_cbd.pdf", bbox_inches="tight", pad_inches=0)

fig, axs = visualize_batch(crd.unsqueeze(0), cmap="coolwarm")
fig.savefig(f"./paper/figures/{image_name}_crd.pdf", bbox_inches="tight", pad_inches=0)

# Patchify
patch_size = p, q = (64, 64)
image_size = image.shape[-2:]

h, w = yd.shape[-2:]
grid_size = h // p, w // q
yp = rearrange(yd, "c (h p) (w q) -> (h w) c p q", p=p, q=q)
fig, axs = visualize_batch(
    yp,
    figsize=(w // q, h // p),
    grid_size=(h // p, w // q),
    cmap="gray",
    norm=Normalize(vmin=yd.min(), vmax=yd.max()),
)
fig.savefig(f"./paper/figures/{image_name}_yp.pdf", bbox_inches="tight", pad_inches=0)

h, w = cbd.shape[-2:]
grid_size = h // p, w // q
cbp = rearrange(cbd, "c (h p) (w q) -> (h w) c p q", p=p, q=q)
fig, axs = visualize_batch(
    cbp,
    figsize=(w // q, h // p),
    grid_size=(h // p, w // q),
    cmap="coolwarm",
    norm=Normalize(vmin=cbd.min(), vmax=cbd.max()),
)
fig.savefig(f"./paper/figures/{image_name}_cbp.pdf", bbox_inches="tight", pad_inches=0)

h, w = crd.shape[-2:]
grid_size = h // p, w // q
crp = rearrange(crd, "c (h p) (w q) -> (h w) c p q", p=p, q=q)
fig, axs = visualize_batch(
    crp,
    figsize=(w // q, h // p),
    grid_size=(h // p, w // q),
    cmap="coolwarm",
    norm=Normalize(vmin=crd.min(), vmax=crd.max()),
)
fig.savefig(f"./paper/figures/{image_name}_crp.pdf", bbox_inches="tight", pad_inches=0)


#### IMF
ycbcr = lrf.rgb_to_ycbcr(image)
y, cb, cr = lrf.chroma_downsampling(ycbcr, scale_factor=(0.5, 0.5), mode="area")

patch_size = (8, 8)
x = lrf.patchify(y, patch_size=patch_size)

# Y
rank = 4
imf = lrf.IMF(rank=rank, bounds=(-16, 15), num_iters=10)
u, v = imf.decompose(x.unsqueeze(0))
u, v = u.squeeze(0), v.squeeze(0)

u, v = lrf.depatchify_uv(u, v, y.shape[-2:], patch_size)
for r in range(rank):
    fig, axs = visualize_batch(u[r].unsqueeze(0), cmap="gray")
    fig.savefig(
        f"./paper/figures/{image_name}_u_y_{r}.pdf", bbox_inches="tight", pad_inches=0
    )

    fig, axs = visualize_batch(v[r].unsqueeze(0), cmap="gray")
    fig.savefig(
        f"./paper/figures/{image_name}_v_y_{r}.pdf", bbox_inches="tight", pad_inches=0
    )

# Cb
x = lrf.patchify(cb, patch_size=patch_size)

rank = 4
imf = lrf.IMF(rank=rank, bounds=(-16, 15), num_iters=10)
u, v = imf.decompose(x.unsqueeze(0))
u, v = u.squeeze(0), v.squeeze(0)

u, v = lrf.depatchify_uv(u, v, cb.shape[-2:], patch_size)
for r in range(rank):
    fig, axs = visualize_batch(u[r].unsqueeze(0), cmap="coolwarm")
    fig.savefig(
        f"./paper/figures/{image_name}_u_cb_{r}.pdf", bbox_inches="tight", pad_inches=0
    )

    fig, axs = visualize_batch(v[r].unsqueeze(0), cmap="coolwarm")
    fig.savefig(
        f"./paper/figures/{image_name}_v_cb_{r}.pdf", bbox_inches="tight", pad_inches=0
    )


# Cr
x = lrf.patchify(cr, patch_size=patch_size)

rank = 4
imf = lrf.IMF(rank=rank, bounds=(-16, 15), num_iters=10)
u, v = imf.decompose(x.unsqueeze(0))
u, v = u.squeeze(0), v.squeeze(0)

u, v = lrf.depatchify_uv(u, v, cr.shape[-2:], patch_size)
for r in range(rank):
    fig, axs = visualize_batch(u[r].unsqueeze(0), cmap="coolwarm")
    fig.savefig(
        f"./paper/figures/{image_name}_u_cr_{r}.pdf", bbox_inches="tight", pad_inches=0
    )

    fig, axs = visualize_batch(v[r].unsqueeze(0), cmap="coolwarm")
    fig.savefig(
        f"./paper/figures/{image_name}_v_cr_{r}.pdf", bbox_inches="tight", pad_inches=0
    )
