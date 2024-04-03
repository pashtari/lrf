from typing import Tuple, Optional, Dict
import math

import torch
from torch import Tensor
import torchvision.transforms.v2.functional as F
from einops import rearrange

from .utils import (
    pad_image,
    unpad_image,
    quantize,
    dequantize,
    dict_to_bytes,
    bytes_to_dict,
    combine_bytes,
    separate_bytes,
    encode_tensors,
    decode_tensors,
)

#### SVD compression ####


def svd_rank(size: Tuple[int, int], compression_ratio: float) -> int:
    """Calculate the rank for SVD based on the compression ratio."""

    num_rows, num_cols = size
    df_input = num_rows * num_cols  # Degrees of freedom of the input matrix
    df_lowrank = num_rows + num_cols  # Degrees of freedom of the low-rank matrix
    rank = max(math.floor(df_input / (compression_ratio * df_lowrank)), 1)
    return rank


def svd_compression_ratio(size: Tuple[int, int], rank: int) -> float:
    """Calculate the compression ratio for SVD based on the rank."""

    num_rows, num_cols = size
    df_input = num_rows * num_cols  # Degrees of freedom of the input matrix
    df_lowrank = rank * (num_rows + num_cols)  # Degrees of freedom of the low-rank matrix
    compression_ratio = df_input / df_lowrank
    return compression_ratio


def svd_encode(
    image: Tensor,
    rank: Optional[int] = None,
    quality: Optional[float] = None,
    compression_ratio: Optional[float] = None,
    bpp: Optional[float] = None,
    dtype: torch.dtype = None,
) -> Tuple[Tensor, Tensor]:
    """Compress an input image using Singular Value Decomposition (SVD)."""

    assert (rank, quality, compression_ratio, bpp) != (
        None,
        None,
        None,
        None,
    ), "Either 'rank', 'compression_ratio', or 'bpp' must be specified."

    if rank is None:
        if quality is not None:
            assert quality >= 0 and quality <= 1, "'quality' must be between 0 and 1."
            rank = max(round(min(image.shape[-2:]) * quality), 1)
        elif compression_ratio is not None:
            rank = svd_rank(image.shape[-2:], compression_ratio)
        else:
            compression_ratio = 8 * image.element_size() * image.shape[0] / bpp
            rank = svd_rank(image.shape[-2:], compression_ratio)

    dtype = image.dtype if dtype is None else dtype

    image = F.to_dtype(image, dtype=torch.float32, scale=True)

    u, s, v = torch.linalg.svd(image, full_matrices=False)
    u, s, v = u[..., :rank], s[..., :rank], v[..., :rank, :]

    u = torch.einsum("...ir, ...r -> ...ir", u, torch.sqrt(s))
    v = torch.einsum("...r, ...rj -> ...rj", torch.sqrt(s), v)

    if not dtype.is_floating_point:
        u, *qtz_u = quantize(u, target_dtype=dtype)
        v, *qtz_v = quantize(v, target_dtype=dtype)
    else:
        qtz_u, qtz_v = (None, None)

    metadata = {"quantization": {"u": qtz_u, "v": qtz_v}}
    encoded_metadata = dict_to_bytes(metadata)

    factors = (u, v)
    encoded_factors = encode_tensors(factors)

    encoded_image = combine_bytes(encoded_metadata, encoded_factors)
    return encoded_image


def svd_decode(encoded_image: bytes, dtype: torch.dtype = torch.uint8) -> Tensor:
    """Decompress an Patch SVD-enocded image."""
    encoded_metadata, encoded_factors = separate_bytes(encoded_image)

    metadata = bytes_to_dict(encoded_metadata)
    qtz = metadata["quantization"]
    qtz_u, qtz_v = qtz["u"], qtz["v"]

    u, v = decode_tensors(encoded_factors)

    u = u if qtz_u is None else dequantize(u, *qtz_u)
    v = v if qtz_v is None else dequantize(v, *qtz_v)

    x = u @ v
    x = F.to_dtype(x.clamp(0, 1), dtype=dtype, scale=True)
    return x


#### Patch SVD compression ####


def patch_svd_patchify(x: Tensor, patch_size: Tuple[int, int] = (8, 8)) -> Tensor:
    """Splits an input image into patches."""

    p, q = patch_size
    patches = rearrange(x, "c (h p) (w q) -> (h w) (c p q)", p=p, q=q)
    return patches


def patch_svd_depatchify_uv(
    u: Tensor, v: Tensor, size: Tuple[int, int], patch_size: Tuple[int, int] = (8, 8)
) -> Tuple[Tensor, Tensor]:
    """Reshape the u and v matrices into their original spatial dimensions."""

    p, q = patch_size
    u_new = rearrange(u, "(h w) r -> 1 r h w", h=size[0] // p)
    v_new = rearrange(v, "(c p q) r -> c r p q", p=p, q=q)
    return u_new, v_new


def patch_svd_depatchify(
    x: Tensor, size: Tuple[int, int], patch_size: Tuple[int, int] = (8, 8)
) -> Tensor:
    """Reconstruct the original image from its patches."""

    p, q = patch_size
    patches = rearrange(x, "(h w) (c p q) -> c (h p) (w q)", p=p, q=q, h=size[0] // p)
    return patches


def patch_svd_encode(
    image: Tensor,
    rank: Optional[int] = None,
    quality: Optional[float] = None,
    compression_ratio: Optional[float] = None,
    bpp: Optional[float] = None,
    patch_size: Tuple[int, int] = (8, 8),
    dtype: torch.dtype = None,
) -> Dict:
    """Compress an input image using Patch SVD."""

    assert (rank, quality, compression_ratio, bpp) != (
        None,
        None,
        None,
        None,
    ), "Either 'rank', 'compression_ratio', or 'bpp' must be specified."

    dtype = image.dtype if dtype is None else dtype

    orig_size = image.shape[-2:]
    image_padded = pad_image(image, patch_size, mode="reflect")
    padded_size = image_padded.shape[-2:]
    element_bytes = image_padded.element_size()
    image_padded = F.to_dtype(image_padded, dtype=torch.float32, scale=True)

    patches = patch_svd_patchify(image_padded, patch_size)

    if rank is None:
        if quality is not None:
            assert quality >= 0 and quality <= 1, "'quality' must be between 0 and 1."
            rank = max(round(min(patches.shape[-2:]) * quality), 1)
        elif compression_ratio is not None:
            rank = svd_rank(patches.shape[-2:], compression_ratio)
        else:
            compression_ratio = 8 * element_bytes * image_padded.shape[0] / bpp
            rank = svd_rank(patches.shape[-2:], compression_ratio)

    u, s, v = torch.linalg.svd(patches, full_matrices=False)
    u, s, v = u[..., :, :rank], s[..., :rank], v[..., :rank, :]

    u = torch.einsum("...ir, ...r -> ...ir", u, torch.sqrt(s))
    v = torch.einsum("...r, ...rj -> ...rj", torch.sqrt(s), v)

    if not dtype.is_floating_point:
        u, *qtz_u = quantize(u, target_dtype=dtype)
        v, *qtz_v = quantize(v, target_dtype=dtype)
    else:
        qtz_u, qtz_v = (None, None)

    metadata = {
        "original size": orig_size,
        "padded size": padded_size,
        "patch size": patch_size,
        "quantization": {"u": qtz_u, "v": qtz_v},
    }
    encoded_metadata = dict_to_bytes(metadata)

    factors = (u, v)
    encoded_factors = encode_tensors(factors)

    encoded_image = combine_bytes(encoded_metadata, encoded_factors)
    return encoded_image


def patch_svd_decode(encoded_image: bytes, dtype: torch.dtype = torch.uint8) -> Tensor:
    """Decompress an Patch SVD-enocded image."""

    encoded_metadata, encoded_factors = separate_bytes(encoded_image)

    metadata = bytes_to_dict(encoded_metadata)
    orig_size = metadata["original size"]
    padded_size = metadata["padded size"]
    patch_size = metadata["patch size"]
    qtz = metadata["quantization"]
    qtz_u, qtz_v = qtz["u"], qtz["v"]

    u, v = decode_tensors(encoded_factors)

    u = u if qtz_u is None else dequantize(u, *qtz_u)
    v = v if qtz_v is None else dequantize(v, *qtz_v)

    reconstructed_patches = u @ v
    reconstructed_image = patch_svd_depatchify(
        reconstructed_patches, padded_size, patch_size
    )
    reconstructed_image = F.to_dtype(
        reconstructed_image.clamp(0, 1), dtype=dtype, scale=True
    )
    reconstructed_image = unpad_image(reconstructed_image, orig_size)
    return reconstructed_image
