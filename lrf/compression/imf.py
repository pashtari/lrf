from typing import Optional, Iterable
import math

import torch
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


def imf_rank(size: tuple[int, int], com_ratio: float) -> int:
    """Calculate the rank for IMF based on the compression ratio.

    Args:
        size (tuple[int, int]): The size the input image (height, width).
        com_ratio (float): The desired compression ratio.

    Returns:
        int: The calculated rank.
    """

    num_rows, num_cols = size
    df_input = num_rows * num_cols  # Degrees of freedom of the input matrix
    df_lowrank = num_rows + num_cols  # Degrees of freedom of the low-rank matrix
    rank = max(math.floor(df_input / (com_ratio * df_lowrank)), 1)
    return rank


def patchify(x: torch.Tensor, patch_size: tuple[int, int]) -> torch.Tensor:
    """Splits an input image into flattened patches.

    Args:
        x (torch.Tensor): The input image tensor of shape (channel, height, width).
        patch_size (tuple[int, int]): The patch size.

    Returns:
        torch.Tensor: The patched image tensor.
    """

    p, q = patch_size
    patches = rearrange(x, "c (h p) (w q) -> (h w) (c p q)", p=p, q=q)
    return patches


def depatchify(
    x: torch.Tensor, size: tuple[int, int], patch_size: tuple[int, int]
) -> torch.Tensor:
    """Reconstruct the original image from its patches.

    Args:
        x (torch.Tensor): The patched image tensor.
        size (tuple[int, int]): The original size of the image (height, width).
        patch_size (tuple[int, int]): The patch size.

    Returns:
        torch.Tensor: The reconstructed image tensor.
    """

    p, q = patch_size
    patches = rearrange(x, "(h w) (c p q) -> c (h p) (w q)", p=p, q=q, h=size[0] // p)
    return patches


def patchify_uv(u: torch.Tensor, v: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Flatten spatial dimensions of the u and v factor maps.

    Args:
        u (torch.Tensor): The u factor map (components map).
        v (torch.Tensor): The v factor map (coefficients map).

    Returns:
        tuple[torch.Tensor, torch.Tensor]: The factor matrices, i.e., the flattened u and v factor maps.
    """

    u_new = rearrange(u, "r 1 h w -> (h w) r")
    v_new = rearrange(v, "r c p q -> (c p q) r")
    return u_new, v_new


def depatchify_uv(
    u: torch.Tensor, v: torch.Tensor, size: tuple[int, int], patch_size: tuple[int, int]
) -> tuple[torch.Tensor, torch.Tensor]:
    """Reshape the u and v matrices into their original spatial dimensions.

    Args:
        u (torch.Tensor): The u factor matrix (patch, rank).
        v (torch.Tensor): The v factor matrix (patch_element, rank).
        size (tuple[int, int]): The original size of the image (height, width).
        patch_size (tuple[int, int]): The patch size.

    Returns:
        tuple[torch.Tensor, torch.Tensor]: The factor maps, i.e., the reshaped u and v factors.
    """

    p, q = patch_size
    u_new = rearrange(u, "(h w) r -> r 1 h w", h=size[0] // p)
    v_new = rearrange(v, "(c p q) r -> r c p q", p=p, q=q)
    return u_new, v_new


def imf_encode(
    image: torch.Tensor,
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
    """
    IMF compression of an image.

    Args:
        image (torch.Tensor): The input image tensor.
        rank (int or tuple[int, int, int], optional): The rank for IMF (default: None).
        quality (float or tuple[float, float, float], optional): The quality for IMF (default: None).
        color_space (str, optional): The color space of the image ('RGB' or 'YCbCr', default: 'YCbCr').
        scale_factor (tuple[float, float], optional): The scale factor for chroma downsampling (default: (0.5, 0.5)).
        patch (bool, optional): Whether to use patch-based encoding (default: True).
        patch_size (tuple[int, int], optional): The patch size (default: (8, 8)).
        bounds (tuple[float, float], optional): The bounds for IMF (default: (-16, 15)).
        dtype (torch.dtype, optional): The data type for encoding (default: torch.int8).
        **kwargs: Additional arguments for IMF decomposition.

    Returns:
        bytes: The encoded image in bytes.
    """

    assert (rank, quality) != (
        None,
        None,
    ), "Either 'rank' or 'quality' must be specified."

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
        if patch:
            image = image.float()

            x = pad_image(image, patch_size, mode="reflect")
            padded_size = x.shape[-2:]
            x = patchify(x, patch_size)

            if rank is None:
                assert (
                    quality >= 0 and quality <= 100
                ), "'quality' must be between 0 and 100."
                R = max(round(min(x.shape[-2:]) * quality / 100), 1)
            else:
                R = rank

            metadata.update(
                {
                    "patch size": patch_size,
                    "original size": image.shape[-2:],
                    "padded size": padded_size,
                    "rank": R,
                }
            )

            imf = IMF(rank=R, bounds=bounds, factor=(0, 1), **kwargs)
            u, v, _ = imf.decompose(x.unsqueeze(0))
            u, v = u.squeeze(0), v.squeeze(0)

            factors = u.to(dtype), v.to(dtype)

        else:
            x = image.float()

            if rank is None:
                assert (
                    quality >= 0 and quality <= 100
                ), "'quality' must be between 0 and 100."
                R = max(round(min(x.shape[-2:]) * quality / 100), 1)
            else:
                R = rank

            metadata["rank"] = R

            imf = IMF(rank=R, bounds=bounds, factor=(0, 1), **kwargs)
            u, v, _ = imf.decompose(x.unsqueeze(0))
            u, v = u.squeeze(0), v.squeeze(0)

            factors = u.to(dtype), v.to(dtype)

    else:  # color_space == "YCbCr"
        if not isinstance(rank, Iterable):
            if rank is None:
                rank = (None, None, None)
            else:
                rank = (rank, max(rank // 2, 1), max(rank // 2, 1))

        if not isinstance(quality, Iterable):
            if quality is None:
                quality = (None, None, None)
            else:
                quality = (quality, quality / 2, quality / 2)

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
                    ), "'quality' must be between 0 and 100."
                    R = max(round(min(x.shape[-2:]) * quality[i] / 100), 1)
                else:
                    R = rank[i]

                metadata["original size"].append(channel.shape[-2:])
                metadata["padded size"].append(padded_size)
                metadata["rank"].append(R)

                imf = IMF(rank=R, bounds=bounds, factor=(0, 1), **kwargs)
                u, v, _ = imf.decompose(x.unsqueeze(0))
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
                    ), "'quality' must be between 0 and 100."
                    R = max(round(min(channel.shape[-2:]) * quality[i] / 100), 1)
                else:
                    R = rank[i]

                metadata["original size"].append(channel.shape[-2:])
                metadata["rank"].append(R)

                imf = IMF(rank=R, bounds=bounds, factor=(0, 1), **kwargs)
                u, v, _ = imf.decompose(channel.unsqueeze(0))
                u, v = u.squeeze(0), v.squeeze(0)

                u, v = u.to(dtype), v.to(dtype)

                factors.extend([u, v])

    encoded_metadata = dict_to_bytes(metadata)
    encoded_factors = combine_bytes([encode_tensor(factor) for factor in factors])
    encoded_image = combine_bytes([encoded_metadata, encoded_factors])

    return encoded_image


def imf_decode(encoded_image: bytes) -> torch.Tensor:
    """
    Decode an IMF-compressed image.

    Args:
        encoded_image (bytes): The encoded image in bytes.

    Returns:
        torch.Tensor: The decoded image tensor.
    """

    encoded_metadata, encoded_factors = separate_bytes(encoded_image, 2)
    metadata = bytes_to_dict(encoded_metadata)

    if metadata["color space"] == "RGB":
        encoded_u, encoded_v = separate_bytes(encoded_factors, 2)
        u, v = decode_tensor(encoded_u), decode_tensor(encoded_v)

        u, v = u.float(), v.float()
        x = IMF.reconstruct(u, v)

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
                x = IMF.reconstruct(u, v)
                channel = depatchify(
                    x, metadata["padded size"][i], metadata["patch size"]
                )
                channel = unpad_image(channel, metadata["original size"][i])
                ycbcr.append(channel)
        else:
            ycbcr = []
            for i, (u, v) in enumerate(((u_y, v_y), (u_cb, v_cb), (u_cr, v_cr))):
                u, v = u.float(), v.float()
                x = IMF.reconstruct(u, v)
                ycbcr.append(x)

        image = chroma_upsampling(
            ycbcr, size=metadata["original size"][0], mode="nearest"
        )
        image = ycbcr_to_rgb(image)

    image = to_dtype(image, getattr(torch, metadata["dtype"]))

    return image
