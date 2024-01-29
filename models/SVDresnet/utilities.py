import math
import torch
from einops import rearrange


def get_svd_components(x, compression_ratio=None):

    H, W = x.shape[-2:]
    R = math.floor(H * W // (compression_ratio * (H + W)))

    u, s, vt = torch.linalg.svd(x, full_matrices=False)
    components = torch.einsum(
        "b c h r, b c r, b c r w -> b r c h w", u[..., :, :R], s[..., :R], vt[..., :R, :]
    )
    return components


def get_svd_patches_components(x, compression_ratio=None, patch_size=8):
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
    # Rearrange image to form columns of flattened patches
    patches = rearrange(
        x, "b c (h p1) (w p2) -> b (h w) (c p1 p2)", p1=patch_size, p2=patch_size
    )

    M, N = patches.shape[-2:]
    R = math.floor(M * N // (compression_ratio * (M + N)))

    u, _, _ = torch.linalg.svd(patches, full_matrices=False)
    u_reshaped = rearrange(
        u[:, :, :R],
        "b (h w) r -> (b r) () h w",
        h = x.shape[-2] // patch_size,
    )
    return u_reshaped


def get_svd_patches_v(x, compression_ratio=1, patch_size=8):
    # Rearrange image to form columns of flattened patches
    patches = rearrange(
        x, "b c (h p1) (w p2) -> b (h w) (c p1 p2)", p1=patch_size, p2=patch_size
    )

    M, N = patches.shape[-2:]
    R = math.floor(M * N // (compression_ratio * (M + N)))

    _, _, vt = torch.linalg.svd(patches, full_matrices=False)
    v_reshaped = rearrange(
        vt[:, :R, :],
        "b r (c p1 p2) -> (b r) (c p1 p2)",
        p1=patch_size,
        p2=patch_size,
    )
    return v_reshaped
