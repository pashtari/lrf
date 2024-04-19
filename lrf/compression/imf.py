from typing import Tuple, Optional, Dict
import math

import torch
from torch import Tensor
from torch.nn.modules.utils import _triple
from einops import rearrange

from lrf.factorization import IMF
from lrf.compression.utils import (
    rgb_to_ycbcr,
    ycbcr_to_rgb,
    chroma_downsampling,
    chroma_upsampling,
    pad_image,
    unpad_image,
    to_dtype,
    dict_to_bytes,
    bytes_to_dict,
    combine_bytes,
    separate_bytes,
    encode_tensors,
    decode_tensors,
)


def imf_rank(size: Tuple[int, int], compression_ratio: float) -> int:
    """Calculate the rank for IMF based on the compression ratio."""

    num_rows, num_cols = size
    df_input = num_rows * num_cols  # Degrees of freedom of the input matrix
    df_lowrank = num_rows + num_cols  # Degrees of freedom of the low-rank matrix
    rank = max(math.floor(df_input / (compression_ratio * df_lowrank)), 1)
    return rank


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


def imf_encode(
    image: Tensor,
    rank: Optional[int | tuple[int, int, int]] = None,
    quality: Optional[float | tuple[float, float, float]] = None,
    color_space="YCbCr",
    scale_factor: tuple[float, float] = (0.5, 0.5),
    patch: bool = True,
    patch_size: Tuple[int, int] = (8, 8),
    dtype: torch.dtype = None,
    **kwargs,
) -> Dict:
    """Compress an input image using Patch IMF (RGB)."""

    assert color_space in (
        "RGB",
        "YCbCr",
    ), "`color_space` must be one of 'RGB' or 'YCbCr'."

    dtype = image.dtype if dtype is None else dtype

    metadata = {
        "dtype": str(image.dtype).split(".")[-1],
        "color space": color_space,
        "patch": patch,
    }

    if color_space == "RGB":
        assert (rank, quality) != (
            None,
            None,
        ), "Either 'rank' or 'quality' for each channel must be specified."

        x = image.float()
        if patch:
            x = pad_image(x, patch_size, mode="reflect")
            padded_size = x.shape[-2:]
            x = patchify(x, patch_size)
            # x =  x * 10
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

        metadata["rank"] = rank

        imf = IMF(rank=rank, **kwargs)
        u, v = imf.decompose(x.unsqueeze(0))
        factors = u.squeeze(0).to(dtype), v.squeeze(0).to(dtype)

    else:  # color_space == "YCbCr"
        quality = _triple(quality)
        rank = _triple(rank)

        for R, Q in zip(rank, quality):
            assert (R, Q) != (
                None,
                None,
            ), "Either 'rank' or 'quality' for each channel must be specified."

        ycbcr = rgb_to_ycbcr(image)
        y, cb, cr = chroma_downsampling(ycbcr, scale_factor=scale_factor)
        factors = []
        for i, channel in enumerate((y, cb, cr)):
            if patch:
                x = pad_image(channel, patch_size, mode="reflect")
                padded_size = x.shape[-2:]

                x = patchify(x, patch_size)
                # x = x * 10

                metadata["patch size"] = patch_size
                if i == 0:
                    metadata["original size"] = [channel.shape[-2:]]
                    metadata["padded size"] = [padded_size]
                else:
                    metadata["original size"].append(channel.shape[-2:])
                    metadata["padded size"].append(padded_size)

            if rank[i] is None:
                assert (
                    quality[i] >= 0 and quality[i] <= 100
                ), "'quality' must be between 0 and 1."
                R = max(round(min(x.shape[-2:]) * quality[i] / 100), 1)

            imf = IMF(rank=R, **kwargs)
            u, v = imf.decompose(x.unsqueeze(0))
            u, v = u.squeeze(0).to(dtype), v.squeeze(0).to(dtype)

            factors.extend([u, v])

    encoded_metadata = dict_to_bytes(metadata)

    encoded_factors = encode_tensors(factors)

    encoded_image = combine_bytes(encoded_metadata, encoded_factors)

    return encoded_image


def imf_decode(encoded_image: bytes) -> Tensor:
    """Decompress an IMF-enocded image."""

    encoded_metadata, encoded_factors = separate_bytes(encoded_image)

    metadata = bytes_to_dict(encoded_metadata)

    if metadata["color space"] == "RGB":
        u, v = decode_tensors(encoded_factors)
        u, v = u.float(), v.float()
        x = u @ v.mT
        # x = x / 10
        if metadata["patch"]:
            image = depatchify(x, metadata["padded size"], metadata["patch size"])
            image = unpad_image(image, metadata["original size"])
        else:
            image = x
    else:  # color_space == "YCbCr"
        u_y, v_y, u_cb, v_cb, u_cr, v_cr = decode_tensors(encoded_factors)
        ycbcr = []
        for i, (u, v) in enumerate(((u_y, v_y), (u_cb, v_cb), (u_cr, v_cr))):
            u, v = u.float(), v.float()
            x = u @ v.mT
            # x = x / 10
            if metadata["patch"]:
                channel = depatchify(
                    x, metadata["padded size"][i], metadata["patch size"]
                )
                ycbcr.append(unpad_image(channel, metadata["original size"][i]))

        image = chroma_upsampling(ycbcr, size=metadata["original size"][0])
        image = ycbcr_to_rgb(image)

    image = to_dtype(image, getattr(torch, metadata["dtype"]))

    return image
