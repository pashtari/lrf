from typing import Optional, Dict
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
    encode_tensor,
    decode_tensor,
)


def imf_rank(size: tuple[int, int], compression_ratio: float) -> int:
    """Calculate the rank for IMF based on the compression ratio."""

    num_rows, num_cols = size
    df_input = num_rows * num_cols  # Degrees of freedom of the input matrix
    df_lowrank = num_rows + num_cols  # Degrees of freedom of the low-rank matrix
    rank = max(math.floor(df_input / (compression_ratio * df_lowrank)), 1)
    return rank


def patchify(x: Tensor, patch_size: tuple[int, int]) -> Tensor:
    """Splits an input image into flattened patches."""

    p, q = patch_size
    patches = rearrange(x, "c (h p) (w q) -> (h w) (c p q)", p=p, q=q)
    return patches


def depatchify(x: Tensor, size: tuple[int, int], patch_size: tuple[int, int]) -> Tensor:
    """Reconstruct the original image from its patches."""

    p, q = patch_size
    patches = rearrange(x, "(h w) (c p q) -> c (h p) (w q)", p=p, q=q, h=size[0] // p)
    return patches


def depatchify_uv(
    u: Tensor, v: Tensor, size: tuple[int, int], patch_size: tuple[int, int]
) -> tuple[Tensor, Tensor]:
    """Reshape the u and v matrices into their original spatial dimensions."""

    p, q = patch_size
    u_new = rearrange(u, "(h w) r -> r 1 h w", h=size[0] // p)
    v_new = rearrange(v, "(c p q) r -> r c p q", p=p, q=q)
    return u_new, v_new


def patchify_uv(u: Tensor, v: Tensor) -> tuple[Tensor, Tensor]:
    """Flatten spatial dimensions of the u and v matrices."""

    u_new = rearrange(u, "r 1 h w -> (h w) r")
    v_new = rearrange(v, "r c p q -> (c p q) r")
    return u_new, v_new


def imf_encode(
    image: Tensor,
    rank: Optional[int | tuple[int, int, int]] = None,
    quality: Optional[float | tuple[float, float, float]] = None,
    color_space: str = "YCbCr",
    scale_factor: tuple[float, float] = (0.5, 0.5),
    patch: bool = True,
    patch_size: tuple[int, int] = (8, 8),
    bounds: tuple[float, float] = (-16, 15),
    dtype: torch.dtype = torch.int8,
    **kwargs,
) -> bytes:
    """Compress an input image using Patch IMF (RGB)."""

    assert color_space in (
        "RGB",
        "YCbCr",
    ), "`color_space` must be one of 'RGB' or 'YCbCr'."

    metadata = {
        "dtype": str(image.dtype).split(".")[-1],
        "color space": color_space,
        "patch": patch,
        "bounds": bounds,
    }

    if color_space == "RGB":
        assert (rank, quality) != (
            None,
            None,
        ), "Either 'rank' or 'quality' must be specified."

        if patch:
            image = image.float()

            x = pad_image(image, patch_size, mode="reflect")
            padded_size = x.shape[-2:]
            x = patchify(x, patch_size)

            if rank is None:
                assert (
                    quality >= 0 and quality <= 100
                ), "'quality' must be between 0 and 100."
                rank = max(round(min(x.shape[-2:]) * quality / 100), 1)

            metadata.update(
                {
                    "patch size": patch_size,
                    "original size": image.shape[-2:],
                    "padded size": padded_size,
                    "rank": rank,
                }
            )

            imf = IMF(rank=rank, bounds=bounds, **kwargs)
            u, v = imf.decompose(x.unsqueeze(0))
            u, v = u.squeeze(0), v.squeeze(0)

            factors = u.to(dtype), v.to(dtype)

        else:
            x = image.float()

            if rank is None:
                assert (
                    quality >= 0 and quality <= 100
                ), "'quality' must be between 0 and 100."
                rank = max(round(min(x.shape[-2:]) * quality / 100), 1)

            metadata["rank"] = rank

            imf = IMF(rank=rank, bounds=bounds, **kwargs)
            u, v = imf.decompose(x.unsqueeze(0))
            u, v = u.squeeze(0), v.squeeze(0)

            factors = u.to(dtype), v.to(dtype)

    else:  # color_space == "YCbCr"
        quality = _triple(quality)
        rank = _triple(rank)

        for R, Q in zip(rank, quality):
            assert (R, Q) != (
                None,
                None,
            ), "Either 'rank' or 'quality' for each channel must be specified."

        image = image.float()

        ycbcr = rgb_to_ycbcr(image)
        y, cb, cr = chroma_downsampling(ycbcr, scale_factor=scale_factor, mode="area")

        if patch:
            metadata["patch size"] = patch_size
            metadata["original size"] = []
            metadata["padded size"] = []
            metadata["rank"] = []
            factors = []
            for i, channel in enumerate((y, cb, cr)):
                x = pad_image(channel, patch_size, mode="reflect")
                padded_size = x.shape[-2:]

                x = patchify(x, patch_size)

                if rank[i] is None:
                    assert (
                        quality[i] >= 0 and quality[i] <= 100
                    ), "'quality' must be between 0 and 1."
                    R = max(round(min(x.shape[-2:]) * quality[i] / 100), 1)

                metadata["original size"].append(channel.shape[-2:])
                metadata["padded size"].append(padded_size)
                metadata["rank"].append(R)

                imf = IMF(rank=R, bounds=bounds, **kwargs)
                u, v = imf.decompose(x.unsqueeze(0))
                u, v = u.squeeze(0), v.squeeze(0)

                u, v = u.to(dtype), v.to(dtype)

                factors.extend([u, v])

        else:
            metadata["original size"] = []
            metadata["rank"] = []
            factors = []
            for i, channel in enumerate((y, cb, cr)):
                if rank[i] is None:
                    assert (
                        quality[i] >= 0 and quality[i] <= 100
                    ), "'quality' must be between 0 and 1."
                    R = max(round(min(channel.shape[-2:]) * quality[i] / 100), 1)

                metadata["original size"].append(channel.shape[-2:])
                metadata["rank"].append(R)

                imf = IMF(rank=R, bounds=bounds, **kwargs)
                u, v = imf.decompose(channel.unsqueeze(0))
                u, v = u.squeeze(0), v.squeeze(0)

                u, v = u.to(dtype), v.to(dtype)

                factors.extend([u, v])

    encoded_metadata = dict_to_bytes(metadata)
    encoded_factors = combine_bytes([encode_tensor(factor) for factor in factors])
    encoded_image = combine_bytes([encoded_metadata, encoded_factors])

    return encoded_image


def imf_decode(encoded_image: bytes) -> Tensor:
    """Decompress an IMF-enocded image."""

    encoded_metadata, encoded_factors = separate_bytes(encoded_image, 2)
    metadata = bytes_to_dict(encoded_metadata)

    if metadata["color space"] == "RGB":
        encoded_u, encoded_v = separate_bytes(encoded_factors, 2)
        u, v = decode_tensor(encoded_u), decode_tensor(encoded_v)

        u, v = u.float(), v.float()
        x = u @ v.mT

        if metadata["patch"]:
            image = depatchify(x, metadata["padded size"], metadata["patch size"])
            image = unpad_image(image, metadata["original size"])

        else:
            image = x

    else:  # color_space == "YCbCr"
        encoded_factors = separate_bytes(encoded_factors, 6)
        u_y, v_y = decode_tensor(encoded_factors[0]), decode_tensor(encoded_factors[1])
        u_cb, v_cb = decode_tensor(encoded_factors[2]), decode_tensor(encoded_factors[3])
        u_cr, v_cr = decode_tensor(encoded_factors[4]), decode_tensor(encoded_factors[5])

        if metadata["patch"]:
            ycbcr = []
            for i, (u, v) in enumerate(((u_y, v_y), (u_cb, v_cb), (u_cr, v_cr))):
                u, v = u.float(), v.float()
                x = u @ v.mT
                channel = depatchify(
                    x, metadata["padded size"][i], metadata["patch size"]
                )
                channel = unpad_image(channel, metadata["original size"][i])
                ycbcr.append(channel)
        else:
            ycbcr = []
            for i, (u, v) in enumerate(((u_y, v_y), (u_cb, v_cb), (u_cr, v_cr))):
                u, v = u.float(), v.float()
                x = u @ v.mT
                ycbcr.append(x)

        image = chroma_upsampling(
            ycbcr, size=metadata["original size"][0], mode="nearest"
        )
        image = ycbcr_to_rgb(image)

    image = to_dtype(image, getattr(torch, metadata["dtype"]))

    return image
