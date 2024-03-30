from typing import Sequence, Tuple, Optional, Union, Dict
from numbers import Real
import math

import torch
from torch import Tensor
import torchvision.transforms.v2.functional as F
from sympy import Symbol, solve
from einops import rearrange

from .metrics import ssim
from ..decomposition.hosvd import hosvd, multi_mode_product, hosvd_rank_feasible_ranges
from .utils import prod, pad_image, unpad_image, quantize, dequantize


#### HOSVD compression ####


def hosvd_rank(size: Tuple[int, int, int], compression_ratio: float) -> int:
    c, h, w = size
    r = Symbol("r", real=True)
    df_input = c * h * w  # Degrees of freedom of the input tensor (image)
    df_core = c * r * r  # Degrees of freedom of the core
    df_factors = c * c + r * h + r * w  # Degrees of freedom of the factors
    rs = solve(df_input - compression_ratio * (df_core + df_factors), r)
    rs = [a.evalf() for a in rs if isinstance(a.evalf(), Real) and a.evalf() > 0]
    r = min(rs)
    r = min(int(math.floor(r)), h, w)
    return c, r, r


def hosvd_compression_ratio(
    size: Sequence[int], rank: Union[int, Sequence[int]]
) -> float:
    df_input = prod(size)  # Degrees of freedom of the input matrix
    df_core = prod(rank)  # Degrees of freedom of the core
    df_factors = sum(
        s * r for s, r in zip(size, rank)
    )  # Degrees of freedom of the factors
    compression_ratio = df_input / (df_core + df_factors)
    return compression_ratio


def hosvd_encode(
    x: Tensor,
    rank: Sequence[int] = None,
    compression_ratio: Optional[float] = None,
    dtype: torch.dtype = None,
) -> Dict:
    """Ecnoder of the Higher-Order Singular Value Decomposition (HOSVD)-based compression."""

    assert (rank is not None) or (
        compression_ratio is not None
    ), "Either 'rank' or 'compression_ratio' must be specified."

    if rank is None:
        rank = hosvd_rank(x.shape, compression_ratio)

    dtype = x.dtype if dtype is None else dtype

    x = F.to_dtype(x, dtype=torch.float32, scale=True)

    core, factors = hosvd(x, rank=rank)

    if dtype != torch.float32:
        core = quantize(core, target_dtype=dtype)
        factors = [quantize(factor, target_dtype=dtype) for factor in factors]

    return {"core": core, "factors": factors}


def hosvd_decode(encoded: Dict, dtype: torch.dtype = torch.uint8) -> Tensor:
    """Decoder of the Higher-Order Singular Value Decomposition (HOSVD)-based compression."""
    core, factors = encoded["core"], encoded["factors"]

    core = dequantize(*core) if isinstance(core, tuple) else core
    factors = [
        dequantize(*factor) if isinstance(factor, tuple) else factor for factor in factors
    ]

    x = multi_mode_product(core, factors, transpose=False)
    x = F.to_dtype(x.clamp(0, 1), dtype=dtype, scale=True)
    return x


#### Patch HOSVD compression ####


def patch_hosvd_tensorize(x: Tensor, patch_size: Tuple[int, int] = (8, 8)) -> Tensor:
    p, q = patch_size
    patches = rearrange(x, "c (h p) (w q) -> (h w) p q c", p=p, q=q)
    return patches


def patch_hosvd_detensorize(
    x: Tensor, size: Tuple[int, int], patch_size: Tuple[int, int] = (8, 8)
) -> Tensor:
    patches = rearrange(x, "(h w) p q c -> c (h p) (w q)", h=size[0] // patch_size[0])
    return patches


def patch_hosvd_optimal_rank(
    x: Tensor, compression_ratio: float, patch_size: Tuple[int, int] = (8, 8)
):
    x = F.to_dtype(x, dtype=torch.float32, scale=True)
    _, h, w = x.shape
    tensor = patch_hosvd_tensorize(x, patch_size)
    n, p, q, c = size = tensor.shape
    rank_ranges = hosvd_rank_feasible_ranges(
        size, compression_ratio, (None, None, None, c)
    )
    (r1_min, r1_max), (_, r2_max), *_ = rank_ranges
    df_input = prod(size)  # Degrees of freedom of the input tensor
    core, factors = hosvd(tensor, rank=(r1_max, r2_max, r2_max, c))
    metric_values = {}
    for r1 in range(r1_min, r1_max + 1):
        r2 = Symbol("r2", real=True)
        df_core = r1 * r2 * r2 * c  # Degrees of freedom of the core
        df_factors = r1 * n + r2 * p + r2 * q + c * c  # Degrees of freedom of the factors
        r2s = solve(df_input - compression_ratio * (df_core + df_factors), r2)
        r2s = [a.evalf() for a in r2s if isinstance(a.evalf(), Real) and a.evalf() > 0]
        if len(r2s) == 0:
            continue

        r2 = min(r2s)
        r2 = min(int(math.floor(r2)), p)
        truncated_core = core[:r1, :r2, :r2, :]
        truncated_factors = [
            factors[0][..., :r1],
            factors[1][..., :r2],
            factors[2][..., :r2],
            factors[3],
        ]
        reconstructed_tensor = multi_mode_product(
            truncated_core, truncated_factors, transpose=False
        )
        reconstructed_image = patch_hosvd_detensorize(
            reconstructed_tensor, (h, w), patch_size
        )
        metric_values[(r1, r2)] = ssim(x, reconstructed_image)

    r1, r2 = max(metric_values, key=metric_values.get)
    return r1, r2, r2, c


def patch_hosvd_encode(
    x: Tensor,
    rank: Optional[Tuple[int, int, int, int]] = None,
    compression_ratio: Optional[float] = None,
    bpp: Optional[float] = None,
    patch_size: Tuple[int, int] = (8, 8),
    dtype: torch.dtype = None,
) -> Tuple[Tensor, Tensor]:
    """Ecnoder of the Patch Higher-Order Singular Value Decomposition (HOSVD) compression."""

    assert (rank, compression_ratio, bpp) != (
        None,
        None,
        None,
    ), "Either 'rank', 'compression_ratio', or 'bpp' must be specified."

    dtype = x.dtype if dtype is None else dtype

    orig_size = x.shape[-2:]
    x = pad_image(x, patch_size, mode="reflect")
    padded_size = x.shape[-2:]

    if rank is None:
        if compression_ratio is None:
            compression_ratio = 8 * x.element_size() * x.shape[0] / bpp

        rank = patch_hosvd_optimal_rank(x, compression_ratio, patch_size)

    x = F.to_dtype(x, dtype=torch.float32, scale=True)

    tensor = patch_hosvd_tensorize(x, patch_size)

    core, factors = hosvd(tensor, rank=rank)

    if dtype != torch.float32:
        core = quantize(core, target_dtype=dtype)
        factors = [quantize(factor, target_dtype=dtype) for factor in factors]

    return {
        "core": core,
        "factors": factors,
        "original size": torch.tensor(orig_size, dtype=torch.int16),
        "padded size": torch.tensor(padded_size, dtype=torch.int16),
        "patch size": torch.tensor(patch_size, dtype=torch.uint8),
    }


def patch_hosvd_decode(encoded: Dict, dtype: torch.dtype = torch.uint8) -> Tensor:
    """Decoder of the Patch Higher-Order Singular Value Decomposition (HOSVD) compression."""
    core, factors = encoded["core"], encoded["factors"]
    orig_size = encoded["original size"].tolist()
    padded_size = encoded["padded size"].tolist()
    patch_size = encoded["patch size"].tolist()

    core = dequantize(*core) if isinstance(core, tuple) else core
    factors = [
        dequantize(*factor) if isinstance(factor, tuple) else factor for factor in factors
    ]

    reconstructed_tensor = multi_mode_product(core, factors, transpose=False)
    reconstructed_image = patch_hosvd_detensorize(
        reconstructed_tensor, padded_size, patch_size
    )
    reconstructed_image = F.to_dtype(
        reconstructed_image.clamp(0, 1), dtype=dtype, scale=True
    )
    reconstructed_image = unpad_image(reconstructed_image, orig_size)
    return reconstructed_image
