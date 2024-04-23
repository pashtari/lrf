from typing import Tuple, Optional, Dict
import math

import torch
from torch import Tensor
import torchvision.transforms.v2.functional as FT
from einops import rearrange

from lrf.compression.utils import (
    pad_image,
    unpad_image,
    quantize,
    dequantize,
    dict_to_bytes,
    bytes_to_dict,
    combine_bytes,
    separate_bytes,
    encode_matrix,
    decode_matrix,
)


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


def patchify(x: Tensor, patch_size: Tuple[int, int] = (8, 8)) -> Tensor:
    """Splits an input image into patches."""

    p, q = patch_size
    patches = rearrange(x, "c (h p) (w q) -> (h w) (c p q)", p=p, q=q)
    return patches


def depatchify(
    x: Tensor, size: Tuple[int, int], patch_size: Tuple[int, int] = (8, 8)
) -> Tensor:
    """Reconstruct the original image from its patches."""

    p, q = patch_size
    patches = rearrange(x, "(h w) (c p q) -> c (h p) (w q)", p=p, q=q, h=size[0] // p)
    return patches


def depatchify_uv(
    u: Tensor, v: Tensor, size: Tuple[int, int], patch_size: Tuple[int, int] = (8, 8)
) -> Tuple[Tensor, Tensor]:
    """Reshape the u and v matrices into their original spatial dimensions."""

    p, q = patch_size
    u_new = rearrange(u, "(h w) r -> 1 r h w", h=size[0] // p)
    v_new = rearrange(v, "(c p q) r -> c r p q", p=p, q=q)
    return u_new, v_new


def svd_encode(
    image: Tensor,
    rank: Optional[int] = None,
    quality: Optional[float] = None,
    patch: bool = True,
    patch_size: Tuple[int, int] = (8, 8),
    dtype: torch.dtype = None,
) -> Dict:
    """Compress an input image using SVD."""

    assert (rank, quality) != (
        None,
        None,
    ), "Either 'rank' or 'quality' must be specified."

    dtype = image.dtype if dtype is None else dtype

    metadata = {"dtype": str(image.dtype).split(".")[-1], "patch": patch}

    x = FT.to_dtype(image, dtype=torch.float32, scale=True)

    if patch:
        x = pad_image(x, patch_size, mode="reflect")
        padded_size = x.shape[-2:]
        x = patchify(x, patch_size)
        metadata.update(
            {
                "patch size": patch_size,
                "original size": image.shape[-2:],
                "padded size": padded_size,
            }
        )

    if rank is None:
        assert quality >= 0 and quality <= 100, "'quality' must be between 0 and 100."
        rank = max(round(min(x.shape[-2:]) * quality / 100), 1)

    u, s, v = torch.linalg.svd(x, full_matrices=False)
    u, s, v = u[..., :, :rank], s[..., :rank], v[..., :rank, :]

    u = torch.einsum("...ir, ...r -> ...ir", u, torch.sqrt(s))
    v = torch.einsum("...r, ...rj -> ...rj", torch.sqrt(s), v)

    if not dtype.is_floating_point:
        u, *qtz_u = quantize(u, target_dtype=dtype)
        v, *qtz_v = quantize(v, target_dtype=dtype)
    else:
        qtz_u, qtz_v = (None, None)

    metadata["quantization"] = {"u": qtz_u, "v": qtz_v}
    encoded_metadata = dict_to_bytes(metadata)

    encoded_factors = combine_bytes([encode_matrix(u), encode_matrix(v)])

    encoded_image = combine_bytes([encoded_metadata, encoded_factors])

    return encoded_image


def svd_decode(encoded_image: bytes) -> Tensor:
    """Decompress an SVD-enocded image."""

    encoded_metadata, encoded_factors = separate_bytes(encoded_image)

    metadata = bytes_to_dict(encoded_metadata)

    encoded_u, encoded_v = separate_bytes(encoded_factors)
    u, v = decode_matrix(encoded_u), decode_matrix(encoded_v)

    qtz = metadata["quantization"]
    qtz_u, qtz_v = qtz["u"], qtz["v"]

    u = u if qtz_u is None else dequantize(u, *qtz_u)
    v = v if qtz_v is None else dequantize(v, *qtz_v)

    x = u @ v

    if metadata["patch"]:
        image = depatchify(x, metadata["padded size"], metadata["patch size"])
        image = unpad_image(image, metadata["original size"])
    else:
        image = x

    image = FT.to_dtype(
        image.clamp(0, 1), dtype=getattr(torch, metadata["dtype"]), scale=True
    )
    return image
