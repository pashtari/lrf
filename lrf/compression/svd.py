from typing import Optional, Dict, Iterable
import math

import torch
from einops import rearrange

from lrf.compression.utils import (
    rgb_to_ycbcr,
    ycbcr_to_rgb,
    chroma_downsampling,
    chroma_upsampling,
    pad_image,
    unpad_image,
    to_dtype,
    quantize,
    dequantize,
    dict_to_bytes,
    bytes_to_dict,
    combine_bytes,
    separate_bytes,
    encode_tensor,
    decode_tensor,
)


def svd_rank(size: tuple[int, int], com_ratio: float) -> int:
    """Calculate the rank for SVD based on the compression ratio.

    Args:
        size (tuple[int, int]): The size of the matrix (rows, columns).
        com_ratio (float): The desired compression ratio.

    Returns:
        int: The calculated rank for SVD.
    """

    num_rows, num_cols = size
    df_input = num_rows * num_cols  # Degrees of freedom of the input matrix
    df_lowrank = num_rows + num_cols  # Degrees of freedom of the low-rank matrix
    rank = max(math.floor(df_input / (com_ratio * df_lowrank)), 1)
    return rank


def svd_compression_ratio(size: tuple[int, int], rank: int) -> float:
    """Calculate the compression ratio for SVD based on the rank.

    Args:
        size (tuple[int, int]): The size of the matrix (rows, columns).
        rank (int): The rank used for SVD.

    Returns:
        float: The calculated compression ratio.
    """

    num_rows, num_cols = size
    df_input = num_rows * num_cols  # Degrees of freedom of the input matrix
    df_lowrank = rank * (num_rows + num_cols)  # Degrees of freedom of the low-rank matrix
    com_ratio = df_input / df_lowrank
    return com_ratio


def patchify(x: torch.Tensor, patch_size: tuple[int, int]) -> torch.Tensor:
    """Splits an input image into flattened patches.

    Args:
        x (torch.Tensor): The input image tensor.
        patch_size (tuple[int, int]): The size of the patches (height, width).

    Returns:
        torch.Tensor: The tensor containing image patches.
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
        size (tuple[int, int]): The size of the original image (height, width).
        patch_size (tuple[int, int]): The patch size.

    Returns:
        torch.Tensor: The reconstructed image tensor.
    """

    p, q = patch_size
    patches = rearrange(x, "(h w) (c p q) -> c (h p) (w q)", p=p, q=q, h=size[0] // p)
    return patches


def depatchify_uv(
    u: torch.Tensor, v: torch.Tensor, size: tuple[int, int], patch_size: tuple[int, int]
) -> tuple[torch.Tensor, torch.Tensor]:
    """Reconstruct the original image from its patches.

    Args:
        x (torch.Tensor): The patched image tensor.
        size (tuple[int, int]): The original size of the image (height, width).
        patch_size (tuple[int, int]): The patch size.

    Returns:
        torch.Tensor: The reconstructed image tensor.
    """

    p, q = patch_size
    u_new = rearrange(u, "(h w) r -> r 1 h w", h=size[0] // p)
    v_new = rearrange(v, "(c p q) r -> r c p q", p=p, q=q)
    return u_new, v_new


def svd_encode(
    image: torch.Tensor,
    rank: Optional[int] = None,
    quality: Optional[float | tuple[float, float, float]] = None,
    color_space: str = "RGB",
    scale_factor: tuple[float, float] = (0.5, 0.5),
    patch: bool = True,
    patch_size: tuple[int, int] = (8, 8),
    dtype: torch.dtype = None,
) -> Dict:
    """Compress an input image using SVD.

    Args:
        image (torch.Tensor): The input image tensor.
        rank (Optional[int]): The rank for SVD compression (default: None).
        quality (Optional[Union[float, tuple[float, float, float]]]): The quality for SVD compression (default: None).
        color_space (str, optional): The color space of the image ('RGB' or 'YCbCr', default: 'RGB').
        scale_factor (tuple[float, float], optional): The scale factor for chroma downsampling (default: (0.5, 0.5)).
        patch (bool, optional): Whether to use patch-based encoding (default: True).
        patch_size (tuple[int, int], optional): The patch size (default: (8, 8)).
        dtype (Optional[torch.dtype], optional): The data type of the compressed image (default: None).

    Returns:
        bytes: The compressed image.
    """

    assert (rank, quality) != (
        None,
        None,
    ), "Either 'rank' or 'quality' must be specified."

    dtype = image.dtype if dtype is None else dtype

    metadata = {
        "dtype": str(image.dtype).split(".")[-1],
        "color space": color_space,
        "patch": patch,
    }

    if color_space == "RGB":
        image = image.float()

        if patch:
            x = pad_image(image, patch_size, mode="reflect")
            padded_size = x.shape[-2:]
            x = patchify(x, patch_size)
            metadata.update(
                {
                    "patch size": patch_size,
                    "original size": image.shape[-2:],
                    "padded size": padded_size,
                }
            )
        else:
            x = image

        if rank is None:
            assert quality >= 0 and quality <= 100, "'quality' must be between 0 and 100."
            R = max(round(min(x.shape[-2:]) * quality / 100), 1)
        else:
            R = rank

        u, s, v = torch.linalg.svd(x, full_matrices=False)
        u, s, v = u[..., :, :R], s[..., :R], v[..., :R, :]

        u = torch.einsum("...ir, ...r -> ...ir", u, torch.sqrt(s))
        v = torch.einsum("...r, ...rj -> ...jr", torch.sqrt(s), v)

        if not dtype.is_floating_point:
            u, *qtz_u = quantize(u, target_dtype=dtype)
            v, *qtz_v = quantize(v, target_dtype=dtype)
        else:
            qtz_u, qtz_v = (None, None)

        metadata["quantization"] = {"u": qtz_u, "v": qtz_v}

        factors = [u, v]

    else:
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
            metadata["quantization"] = {"u": [], "v": []}
            factors = []
            for i, channel in enumerate((y, cb, cr)):
                x = pad_image(channel, patch_size, mode="reflect")
                padded_size = x.shape[-2:]

                x = patchify(x, patch_size)

                metadata["padded size"].append(padded_size)

                if rank[i] is None:
                    assert (
                        quality[i] >= 0 and quality[i] <= 100
                    ), "'quality' must be between 0 and 100."
                    R = max(round(min(x.shape[-2:]) * quality[i] / 100), 1)
                else:
                    R = rank

                metadata["original size"].append(channel.shape[-2:])
                metadata["padded size"].append(padded_size)
                metadata["rank"].append(R)

                u, s, v = torch.linalg.svd(x, full_matrices=False)
                u, s, v = u[..., :, :R], s[..., :R], v[..., :R, :]

                u = torch.einsum("...ir, ...r -> ...ir", u, torch.sqrt(s))
                v = torch.einsum("...r, ...rj -> ...jr", torch.sqrt(s), v)

                if not dtype.is_floating_point:
                    u, *qtz_u = quantize(u, target_dtype=dtype)
                    v, *qtz_v = quantize(v, target_dtype=dtype)
                else:
                    qtz_u, qtz_v = (None, None)

                metadata["quantization"]["u"].append(qtz_u)
                metadata["quantization"]["v"].append(qtz_v)

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
                    R = rank

                metadata["original size"].append(channel.shape[-2:])
                metadata["rank"].append(R)

                u, s, v = torch.linalg.svd(channel.squeeze(0), full_matrices=False)
                u, s, v = u[..., :, :R], s[..., :R], v[..., :R, :]

                u = torch.einsum("...ir, ...r -> ...ir", u, torch.sqrt(s))
                v = torch.einsum("...r, ...rj -> ...jr", torch.sqrt(s), v)

                if not dtype.is_floating_point:
                    u, *qtz_u = quantize(u, target_dtype=dtype)
                    v, *qtz_v = quantize(v, target_dtype=dtype)
                else:
                    qtz_u, qtz_v = (None, None)

                metadata["quantization"]["u"].append(qtz_u)
                metadata["quantization"]["v"].append(qtz_v)

                factors.extend([u, v])

    encoded_metadata = dict_to_bytes(metadata)
    encoded_factors = combine_bytes([encode_tensor(factor) for factor in factors])

    encoded_image = combine_bytes([encoded_metadata, encoded_factors])

    return encoded_image


def svd_decode(encoded_image: bytes) -> torch.Tensor:
    """Decompress an SVD-encoded image.

    Args:
        encoded_image (bytes): The encoded image data.

    Returns:
        torch.Tensor: The decompressed image tensor.
    """

    encoded_metadata, encoded_factors = separate_bytes(encoded_image, 2)
    metadata = bytes_to_dict(encoded_metadata)

    if metadata["color space"] == "RGB":
        encoded_u, encoded_v = separate_bytes(encoded_factors, 2)
        u, v = decode_tensor(encoded_u), decode_tensor(encoded_v)

        qtz = metadata["quantization"]
        qtz_u, qtz_v = qtz["u"], qtz["v"]

        u = u if qtz_u is None else dequantize(u, *qtz_u)
        v = v if qtz_v is None else dequantize(v, *qtz_v)

        x = u @ v.mT

        if metadata["patch"]:
            image = depatchify(x, metadata["padded size"], metadata["patch size"])
            image = unpad_image(image, metadata["original size"])
        else:
            image = x

    else:
        encoded_factors = separate_bytes(encoded_factors, 6)

        u_y, v_y = decode_tensor(encoded_factors[0]), decode_tensor(encoded_factors[1])
        u_cb, v_cb = decode_tensor(encoded_factors[2]), decode_tensor(encoded_factors[3])
        u_cr, v_cr = decode_tensor(encoded_factors[4]), decode_tensor(encoded_factors[5])

        ycbcr = []
        for i, (u, v) in enumerate(((u_y, v_y), (u_cb, v_cb), (u_cr, v_cr))):

            qtz = metadata["quantization"]
            qtz_u, qtz_v = qtz["u"][i], qtz["v"][i]

            u = u if qtz_u is None else dequantize(u, *qtz_u)
            v = v if qtz_v is None else dequantize(v, *qtz_v)

            x = u @ v.mT

            if metadata["patch"]:
                channel = depatchify(
                    x, metadata["padded size"][i], metadata["patch size"]
                )
                channel = unpad_image(channel, metadata["original size"][i])
            else:
                channel = x

            ycbcr.append(channel)

        image = chroma_upsampling(ycbcr, size=metadata["original size"][0], mode="area")
        image = ycbcr_to_rgb(image)

    image = to_dtype(image, getattr(torch, metadata["dtype"]))

    return image
