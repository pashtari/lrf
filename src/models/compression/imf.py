from typing import Tuple, Optional, Dict
import math

import torch
from torch import Tensor
from torch.nn.modules.utils import _triple
from einops import rearrange

from ..decomposition import IMF
from .utils import (
    rgb_to_ycbcr,
    ycbcr_to_rgb,
    chroma_downsampling,
    chroma_upsampling,
    pad_image,
    unpad_image,
    to_dtype,
    pack_tensor,
    unpack_tensor,
    dict_to_bytes,
    bytes_to_dict,
    combine_bytes,
    separate_bytes,
    encode_tensors,
    decode_tensors,
)


def imf_rank(size: Tuple[int, int], compression_ratio: float) -> int:
    """Calculate the rank for IMF based on the compression ratio.

    Args:
        size (Tuple[int, int]): _description_
        compression_ratio (float): _description_

    Returns:
        int: The calculated rank.
    """

    num_rows, num_cols = size
    df_input = num_rows * num_cols  # Degrees of freedom of the input matrix
    df_lowrank = num_rows + num_cols  # Degrees of freedom of the low-rank matrix
    rank = max(math.floor(df_input / (compression_ratio * df_lowrank)), 1)
    return rank


def patch_imf_patchify(x: Tensor, patch_size: Tuple[int, int] = (8, 8)) -> Tensor:
    """Splits an input image into patches."""

    p, q = patch_size
    patches = rearrange(x, "c (h p) (w q) -> (h w) (c p q)", p=p, q=q)
    return patches


def patch_imf_depatchify_uv(
    u: Tensor, v: Tensor, size: Tuple[int, int], patch_size: Tuple[int, int] = (8, 8)
) -> Tuple[Tensor, Tensor]:
    """Reshape the u and v matrices into their original spatial dimensions."""

    p, q = patch_size
    u_new = rearrange(u, "(h w) r -> 1 r h w", h=size[0] // p)
    v_new = rearrange(v, "(c p q) r -> c r p q", p=p, q=q)
    return u_new, v_new


def patch_imf_depatchify(
    x: Tensor, size: Tuple[int, int], patch_size: Tuple[int, int] = (8, 8)
) -> Tensor:
    """Reconstruct the original image from its patches."""

    p, q = patch_size
    patches = rearrange(x, "(h w) (c p q) -> c (h p) (w q)", p=p, q=q, h=size[0] // p)
    return patches


def patch_imf_rgb_encode(
    image: Tensor,
    rank: Optional[int] = None,
    quality: Optional[float] = None,
    compression_ratio: Optional[float] = None,
    bpp: Optional[float] = None,
    patch_size: Tuple[int, int] = (8, 8),
    dtype: torch.dtype = None,
    **kwargs,
) -> Dict:
    """Compress an input image using Patch IMF (RGB)."""

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

    patches = patch_imf_patchify(image_padded, patch_size)

    if rank is None:
        if quality is not None:
            assert quality >= 0 and quality <= 1, "'quality' must be between 0 and 1."
            rank = max(round(min(patches.shape[-2:]) * quality), 1)
        elif compression_ratio is not None:
            rank = imf_rank(patches.shape[-2:], compression_ratio)
        else:
            compression_ratio = 8 * element_bytes * image_padded.shape[0] / bpp
            rank = imf_rank(patches.shape[-2:], compression_ratio)

    patches = patches.float()  # * 10

    imf = IMF(rank=rank, **kwargs)
    u, v = imf.decompose(patches.unsqueeze(0))

    u, v = u.squeeze(0).to(dtype), v.squeeze(0).to(dtype)

    metadata = {
        "original size": orig_size,
        "padded size": padded_size,
        "patch size": patch_size,
        "rank": rank,
    }
    encoded_metadata = dict_to_bytes(metadata)

    factors = (u, v)
    encoded_factors = encode_tensors(factors)

    encoded_image = combine_bytes(encoded_metadata, encoded_factors)
    return encoded_image


def patch_imf_rgb_decode(
    encoded_image: bytes, dtype: torch.dtype = torch.uint8
) -> Tensor:
    """Decompress an Patch IMF (RGB)-enocded image."""

    encoded_metadata, encoded_factors = separate_bytes(encoded_image)

    metadata = bytes_to_dict(encoded_metadata)
    orig_size = metadata["original size"]
    padded_size = metadata["padded size"]
    patch_size = metadata["patch size"]

    u, v = decode_tensors(encoded_factors)
    u, v = u.float(), v.float()

    patches = u @ v.transpose(-2, -1)
    # patches = patches / 10

    patches = to_dtype(patches, dtype)
    image = patch_imf_depatchify(patches, padded_size, patch_size)
    image = unpad_image(image, orig_size)
    return image


def patch_imf_ycbcr_encode(
    image: Tensor,
    quality: Optional[float | tuple[float, float, float]] = None,
    scale_factor: tuple[float, float] = (0.5, 0.5),
    patch_size: tuple[int, int] = (8, 8),
    bounds: tuple[None | float, None | float] = (None, None),
    dtype: torch.dtype = None,
    **kwargs,
) -> Dict:
    """Compress an input image using Patch IMF (YCbCr)."""

    quality = _triple(quality)

    dtype = image.dtype if dtype is None else dtype

    ycbcr_img = rgb_to_ycbcr(image)
    y, cb, cr = chroma_downsampling(ycbcr_img, scale_factor=scale_factor)

    metadata = {"patch size": patch_size, "original size": [], "padded size": []}
    factors = []

    for i, channel in enumerate((y, cb, cr)):
        channel_padded = pad_image(channel, patch_size, mode="reflect")
        padded_size = channel_padded.shape[-2:]

        patches = patch_imf_patchify(channel_padded, patch_size)
        # patches = patches * 10

        assert quality[i] >= 0 and quality[i] <= 1, "'quality' must be between 0 and 1."
        rank = max(round(min(patches.shape[-2:]) * quality[i]), 1)

        imf = IMF(rank=rank, bounds=bounds, **kwargs)
        u, v = imf.decompose(patches.unsqueeze(0))

        u, v = u.squeeze(0).to(dtype), v.squeeze(0).to(dtype)

        metadata["original size"].append(channel.shape[-2:])
        metadata["padded size"].append(padded_size)
        factors.extend([u, v])

    encoded_metadata = dict_to_bytes(metadata)

    encoded_factors = encode_tensors(factors)

    encoded_image = combine_bytes(encoded_metadata, encoded_factors)
    return encoded_image


def patch_imf_ycbcr_decode(
    encoded_image: bytes, dtype: torch.dtype = torch.uint8
) -> Tensor:
    """Decompress an Patch IMF (YCbCr)-enocded image."""

    encoded_metadata, encoded_factors = separate_bytes(encoded_image)

    metadata = bytes_to_dict(encoded_metadata)

    u_y, v_y, u_cb, v_cb, u_cr, v_cr = decode_tensors(encoded_factors)

    ycbcr = []
    for i, (u, v) in enumerate(((u_y, v_y), (u_cb, v_cb), (u_cr, v_cr))):
        u, v = u.float(), v.float()
        patches = u @ v.mT
        # patches = patches / 10
        channel = patch_imf_depatchify(
            patches, metadata["padded size"][i], metadata["patch size"]
        )
        ycbcr.append(unpad_image(channel, metadata["original size"][i]))

    ycbcr_img = chroma_upsampling(ycbcr, size=metadata["original size"][0])
    rgb_img = ycbcr_to_rgb(ycbcr_img)
    rgb_img = to_dtype(rgb_img, dtype=dtype)
    return rgb_img


def patch_imf_encode(*args, color_space="RGB", **kwargs):
    assert color_space in (
        "RGB",
        "YCbCr",
    ), "`color_space` must be one of 'RGB' or 'YCbCr'."
    if color_space == "RGB":
        return patch_imf_rgb_encode(*args, **kwargs)
    else:
        return patch_imf_ycbcr_encode(*args, **kwargs)


def patch_imf_decode(*args, color_space="RGB", **kwargs):
    assert color_space in (
        "RGB",
        "YCbCr",
    ), "`color_space` must be one of 'RGB' or 'YCbCr'."
    if color_space == "RGB":
        return patch_imf_rgb_decode(*args, **kwargs)
    else:
        return patch_imf_ycbcr_decode(*args, **kwargs)
