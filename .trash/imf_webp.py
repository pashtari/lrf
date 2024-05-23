
from typing import Optional, Dict
import math

import torch
from torch import Tensor
from torch.nn.modules.utils import _triple
from einops import rearrange

from lrf.factorization import IMF
from lrf.compression import batched_pil_encode, batched_pil_decode
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
    pil_kwargs: Dict = None,
    dtype: torch.dtype = None,
    **kwargs,
) -> bytes:
    """Compress an input image using Patch IMF (RGB)."""

    assert color_space in (
        "RGB",
        "YCbCr",
    ), "`color_space` must be one of 'RGB' or 'YCbCr'."

    pil_kwargs = (
        {"lossless": True, "format": "WEBP", "quality": 100}
        if pil_kwargs is None
        else pil_kwargs
    )

    dtype = image.dtype if dtype is None else dtype

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
            # x =  x * 10

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

            u, v = u - bounds[0], v - bounds[0]
            u, v = to_dtype(u, torch.uint8), to_dtype(v, torch.uint8)

            u, v = depatchify_uv(u, v, padded_size, patch_size)

            encoded_metadata = dict_to_bytes(metadata)
            encoded_factors = combine_bytes(
                [batched_pil_encode(u, **pil_kwargs), batched_pil_encode(v, **pil_kwargs)]
            )

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

            u, v = u.to(dtype), v.to(dtype)

            encoded_metadata = dict_to_bytes(metadata)
            encoded_factors = combine_bytes([encode_tensor(u), encode_tensor(v)])

    elif color_space == "XYCbCr":
        quality = _triple(quality)
        rank = _triple(rank)

        for R, Q in zip(rank, quality):
            assert (R, Q) != (
                None,
                None,
            ), "Either 'rank' or 'quality' for each channel must be specified."

        metadata["original size"] = image.shape[-2:]

        image = image.float()

        ycbcr = rgb_to_ycbcr(image)

        patch_size = (patch_size, (int(patch_size[0] * scale_factor[0]), int(patch_size[1] * scale_factor[0])), (int(patch_size[0] * scale_factor[1]), int(patch_size[1] * scale_factor[1])))
        metadata["patch size"] = patch_size

        ycbcr = pad_image(ycbcr, patch_size[0], mode="reflect")
        

        y, cb, cr = chroma_downsampling(ycbcr, scale_factor=scale_factor, mode="area")

        factors = []
        patch_matrices= []
        metadata["padded size"] = []
        for i, channel in enumerate((y, cb, cr)):
            metadata["padded size"].append(channel.shape[-2:])
            patch_matrices.append(patchify(channel, patch_size[i]))

        x = torch.cat(patch_matrices, dim=-1)

        assert (
            quality >= 0 and quality <= 100
        ), "'quality' must be between 0 and 1."
        R = max(round(min(x.shape[-2:]) * quality / 100), 1)

        metadata["rank"].append(R)

        imf = IMF(rank=R, bounds=bounds, **kwargs)
        u, v = imf.decompose(x.unsqueeze(0))
        u, v = u.squeeze(0), v.squeeze(0)

        u, v = u.to(dtype), v.to(dtype)
        encoded_metadata = dict_to_bytes(metadata)
        encoded_factors = combine_bytes([encode_tensor(u), encode_tensor(v)])

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

                u, v = u - bounds[0], v - bounds[0]
                u, v = to_dtype(u, torch.uint8), to_dtype(v, torch.uint8)

                u, v = depatchify_uv(u, v, padded_size, patch_size)

                factors.extend([u, v])

            encoded_metadata = dict_to_bytes(metadata)
            encoded_factors = combine_bytes(
                [batched_pil_encode(factor, **pil_kwargs) for factor in factors]
            )

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

        if metadata["patch"]:
            u, v = batched_pil_decode(
                encoded_u, num_images=metadata["rank"]
            ), batched_pil_decode(encoded_v, num_images=metadata["rank"])

            u, v = patchify_uv(u[:, 0:1, ...], v)

            bounds = metadata["bounds"]
            u, v = u.float() + bounds[0], v.float() + bounds[0]

            x = u @ v.mT

            image = depatchify(x, metadata["padded size"], metadata["patch size"])
            image = unpad_image(image, metadata["original size"])

        else:
            u, v = decode_tensor(encoded_u), decode_tensor(encoded_v)

            u, v = u.float(), v.float()

            x = u @ v.mT

            image = x

    elif metadata["color space"] == "XYCbCr":

        (py, qy), (pcb, qcb), (pcr, qcr) = metadata["patch size"]


        encoded_factors = separate_bytes(encoded_factors, 2)

        u, v = decode_tensor(encoded_u), decode_tensor(encoded_v)

        u, v = u.float(), v.float()

        x = u @ v.mT

        x_y, x_cb, x_cr = torch.split(x,split_size_or_sections=[py*qy, pcb*qcb, pcr*qcr], dim=-1)

        ycbcr = []
        for i, x in enumerate((x_y, x_cb, x_cr)):
            ycbcr.append(depatchify(x, metadata["padded size"][i], metadata["patch size"][i]))

        ycbcr = chroma_upsampling(ycbcr, size=ycbcr[0].shape[-2:], mode="area")
        
        ycbcr = unpad_image(ycbcr, metadata["original size"])

        image = ycbcr_to_rgb(ycbcr)


    else:  # color_space == "YCbCr"
        encoded_factors = separate_bytes(encoded_factors, 6)

        if metadata["patch"]:
            R1, R2, R3 = metadata["rank"]
            u_y, v_y = batched_pil_decode(encoded_factors[0], R1), batched_pil_decode(
                encoded_factors[1], R1
            )
            u_cb, v_cb = batched_pil_decode(encoded_factors[2], R2), batched_pil_decode(
                encoded_factors[3], R2
            )
            u_cr, v_cr = batched_pil_decode(encoded_factors[4], R3), batched_pil_decode(
                encoded_factors[5], R3
            )

            ycbcr = []
            for i, (u, v) in enumerate(((u_y, v_y), (u_cb, v_cb), (u_cr, v_cr))):
                u, v = patchify_uv(u[:, 0:1, ...], v[:, 0:1, ...])

                bounds = metadata["bounds"]
                u, v = u.float() + bounds[0], v.float() + bounds[0]

                x = u @ v.mT

                channel = depatchify(
                    x, metadata["padded size"][i], metadata["patch size"]
                )
                ycbcr.append(unpad_image(channel, metadata["original size"][i]))
        else:
            u_y, v_y = decode_tensor(encoded_factors[0]), decode_tensor(
                encoded_factors[1]
            )
            u_cb, v_cb = decode_tensor(encoded_factors[2]), decode_tensor(
                encoded_factors[3]
            )
            u_cr, v_cr = decode_tensor(encoded_factors[4]), decode_tensor(
                encoded_factors[5]
            )

            ycbcr = []
            for i, (u, v) in enumerate(((u_y, v_y), (u_cb, v_cb), (u_cr, v_cr))):
                u, v = u.float(), v.float()
                x = u @ v.mT
                ycbcr.append(x)

        image = chroma_upsampling(ycbcr, size=metadata["original size"][0], mode="area")
        image = ycbcr_to_rgb(image)

    image = to_dtype(image, getattr(torch, metadata["dtype"]))

    return image