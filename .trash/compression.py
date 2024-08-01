from typing import Sequence, Tuple, Optional, Union
from abc import ABC, abstractmethod
from numbers import Real
import math

import torch
from torch import nn, Tensor
import torch.nn.functional as F
from torch.nn.modules.utils import _pair, _triple, _ntuple
import torch_dct as dct
from einops import rearrange
from sympy import Symbol, solve


from .metrics import relative_error, ssim
from .utils import zscore_normalize, prod
from ..decomposition import (
    hosvd_rank_feasible_ranges,
    batched_hosvd,
    batched_multi_mode_product,
    tt_rank_feasible_ranges,
    batched_ttd,
    batched_contract_tt,
)


class Compression(ABC, nn.Module):
    """Abstract base class for compression methods."""

    @abstractmethod
    def encode(self, x: Tensor, *args, **kwargs) -> Union[Tensor, Tuple[Tensor, ...]]:
        """Compress the input tensor."""
        pass

    @abstractmethod
    def decode(self, *args, **kwargs) -> Tensor:
        """Decompress the input tensor."""
        pass

    def loss(self, x: Tensor, *args, **kwargs) -> Tensor:
        """Calculate the loss between the original and decompressed tensors.

        Args:
            x (Tensor): The original tensor.

        Returns:
            Tensor: The calculated loss.
        """
        y = self.decode(*args, **kwargs)
        return relative_error(x, y)

    def forward(self, x: Tensor, *args, **kwargs) -> Tensor:
        """Forward pass through the compression model.

        Args:
            x (Tensor): Input tensor to be compressed and decompressed.

        Returns:
            Tensor: The decompressed tensor.
        """
        compressed = self.encode(x, *args, **kwargs)
        decompressed = self.decode(compressed)
        return decompressed


class Interpolate(Compression):
    """Compression method using interpolation for resizing images."""

    def __init__(self, **kwargs) -> None:
        super().__init__()
        self.interpolation_kwargs = kwargs

    def get_new_size(
        self, original_size: Tuple[int, int], compression_ratio: float
    ) -> Tuple[int, int]:
        """Calculate new size for an image given a compression ratio.

        Args:
            original_size (Tuple[int, int]): The original (height, width) of the image.
            compression_ratio (float): The desired compression ratio.

        Returns:
            Tuple[int, int]: The new (height, width) of the image.
        """
        original_size = _pair(original_size)
        height, width = original_size
        new_height = max(math.floor(height / math.sqrt(compression_ratio)), 1)
        new_width = max(math.floor(width / math.sqrt(compression_ratio)), 1)
        return new_height, new_width

    def get_compression_ratio(
        self, original_size: Tuple[int, int], new_size: Tuple[int, int]
    ) -> float:
        original_size = _pair(original_size)
        new_size = _pair(new_size)
        orig_height, orig_width = original_size
        new_height, new_width = new_size
        compression_ratio = (orig_height * orig_width) / (new_height * new_width)
        return compression_ratio

    def encode(
        self,
        x: Tensor,
        compression_ratio: Optional[float] = None,
        new_size: Optional[Tuple[int, int]] = None,
    ) -> Tensor:
        """Compress the input image by resizing.

        Args:
            x (Tensor): Input image tensor.
            compression_ratio (Optional[float]): Compression ratio to determine new size.
            new_size (Optional[Tuple[int, int]]): Directly specify new size.

        Returns:
            Tensor: The resized (compressed) image tensor.
        """
        assert (
            new_size is not None or compression_ratio is not None
        ), "Either new_size or compression_ratio must be specified."

        original_size = x.shape[-2:]
        if new_size is None:
            new_size = self.get_new_size(original_size, compression_ratio)

        new_size = _pair(new_size)

        resized_image = F.interpolate(x, size=new_size, **self.interpolation_kwargs)
        self.real_compression_ratio = self.get_compression_ratio(original_size, new_size)
        return resized_image

    def decode(self, x: Tensor, original_size: Tuple[int, int]) -> Tensor:
        """Decompress the image by resizing back to the original size.

        Args:
            x (Tensor): Compressed image tensor.
            original_size (Tuple[int, int]): The original size to decompress to.

        Returns:
            Tensor: The decompressed (resized) image tensor.
        """
        decompressed_image = F.interpolate(
            x, size=original_size, **self.interpolation_kwargs
        )
        return decompressed_image

    def forward(self, x: Tensor, *args, **kwargs) -> Tensor:
        # x: B × C × H × W
        z = self.encode(x, *args, **kwargs)
        y = self.decode(z, x.shape[-2:])
        return y


class DCT(Compression):
    """Implements Discrete Cosine Transform (DCT) based compression and decompression."""

    def get_cutoff(
        self, size: Tuple[int, int], compression_ratio: float
    ) -> Tuple[int, int]:
        """Calculate the cutoff frequencies for DCT based on compression ratio.

        Args:
            size: Tuple[int, int], the height and width of the input tensor.
            compression_ratio: float, the desired compression ratio.

        Returns:
            Tuple[int, int], the cutoff frequencies for height and width.
        """
        size = _pair(size)
        height, width = size
        cutoff_height = max(math.floor(height / (math.sqrt(compression_ratio))), 1)
        cutoff_width = max(math.floor(width / math.sqrt(compression_ratio)), 1)
        return (cutoff_height, cutoff_width)

    def get_compression_ratio(
        self, size: Tuple[int, int], cuttoff: Tuple[int, int]
    ) -> float:
        size = _pair(size)
        cuttoff = _pair(cuttoff)
        orig_height, orig_width = size
        new_height, new_width = cuttoff
        compression_ratio = (orig_height * orig_width) / (new_height * new_width)
        return compression_ratio

    def encode(
        self,
        x: Tensor,
        compression_ratio: Optional[float] = None,
        cutoff: Optional[Tuple[int, int]] = None,
    ) -> Tensor:
        """Compress the input tensor using DCT.

        Args:
            x: Tensor, the input tensor to be compressed.
            compression_ratio: Optional[float], the desired compression ratio.
            cutoff: Optional[Tuple[int, int]], the cutoff frequencies for DCT.

        Returns:
            Tensor, the compressed tensor.
        """
        assert (cutoff is not None) or (
            compression_ratio is not None
        ), "Either 'cutoff' or 'compression_ratio' must be specified."

        original_size = x.shape[-2:]
        if cutoff is None:
            cutoff = self.get_cutoff(original_size, compression_ratio)

        cutoff = _pair(cutoff)

        x_dct = dct.dct_2d(x)  # Perform DCT
        x_dct = x_dct[..., : cutoff[0], : cutoff[1]]  # Keep frequencies up to cutoff

        self.real_compression_ratio = self.get_compression_ratio(original_size, cutoff)
        return x_dct

    def decode(
        self, x: Tensor, original_size: Tuple[int, int], pad: bool = True, zscore=False
    ) -> Tensor:
        """Decompress the input tensor using inverse DCT.

        Args:
            x: Tensor, the compressed tensor to be decompressed.
            original_size: Tuple[int, int], the original height and width of the input tensor.
            pad: bool, whether to pad the tensor to its original size.

        Returns:
            Tensor, the decompressed tensor.
        """
        original_size = _pair(original_size)
        height, width = original_size
        if pad:
            cutoff_height, cutoff_width = x.shape[-2:]
            x = F.pad(x, pad=(0, width - cutoff_width, 0, height - cutoff_height))

        y = dct.idct_2d(x)  # Perform inverse DCT
        if zscore:
            y = zscore_normalize(y)
        return y

    def forward(self, x: Tensor, *args, pad=True, zscore=False, **kwargs) -> Tensor:
        # x: B × C × H × W
        z = self.encode(x, *args, **kwargs)
        y = self.decode(z, x.shape[-2:], pad=pad, zscore=zscore)
        return y


class PatchTTD(TTD):
    """Implements TTD-based compression with patchification."""

    def __init__(self, patch_size: Tuple[int, int] = (8, 8)) -> None:
        super().__init__()
        self.patch_size = _pair(patch_size)

    def tensorize(self, x: Tensor) -> Tensor:
        p, q = self.patch_size
        patches = rearrange(x, "b c (h p) (w q) -> b (h w) p q c", p=p, q=q)
        return patches

    def detensorize(self, x: Tensor, size: Tuple[int, int]) -> Tensor:
        p, q = self.patch_size
        patches = rearrange(
            x, "b (h w) p q c -> b c (h p) (w q)", p=p, q=q, h=size[-2] // p
        )
        return patches

    def get_optimal_rank(self, x: Tensor, compression_ratio: float):
        _, h, w = x.shape[1:]
        tensor = self.tensorize(x)
        n, p, q, c = size = tensor.shape[1:]
        rank_ranges = tt_rank_feasible_ranges(size, compression_ratio)
        (r1_min, r1_max), (r2_min, r2_max), _ = rank_ranges
        df_input = prod(size)  # Degrees of freedom of the input tensor
        factors = batched_ttd(tensor, rank=(r1_max, r2_max, c))
        metric_values = {}
        for r1 in range(r1_min, r1_max + 1):
            r2 = Symbol("r2", real=True)
            df_factors = (
                r1 * n + r1 * p * r2 + r2 * q * c + c * c
            )  # Degrees of freedom of the factors
            r2s = solve(df_input - compression_ratio * df_factors, r2)
            r2s = [
                a.evalf() for a in r2s if isinstance(a.evalf(), Real) and a.evalf() > 0
            ]
            if len(r2s) == 0:
                continue

            r2 = min(r2s)
            r2 = max(min(int(math.floor(r2)), r2_min), r2_max)
            truncated_factors = [
                factors[0][..., :, :r1],
                factors[1][..., :r1, :, :r2],
                factors[2][..., :r2, :, :c],
                factors[3][..., :c, :],
            ]
            reconstructed_tensor = batched_contract_tt(truncated_factors)
            reconstructed_image = self.detensorize(reconstructed_tensor, (h, w))
            metric_values[(r1, r2)] = ssim(x, reconstructed_image)

        r1, r2 = max(metric_values, key=metric_values.get)
        return r1, r2, c

    def encode(
        self,
        x: Tensor,
        rank: Tuple[int, int, int, int] = None,
        compression_ratio: Optional[float] = None,
    ) -> Tuple[Tensor, Tensor]:

        assert (rank is not None) or (
            compression_ratio is not None
        ), "Either 'rank' or 'compression_ratio' must be specified."

        tensor = self.tensorize(x)
        original_size = tensor.shape[1:]

        if rank is None:
            rank = self.get_optimal_rank(x, compression_ratio)

        factors = batched_ttd(tensor, rank=rank)

        self.real_compression_ratio = self.get_compression_ratio(
            size=original_size, rank=rank
        )
        return factors

    def decode(self, factors: Sequence[Tensor], size: Tuple[int, int]) -> Tensor:
        reconstructed_tensor = super().decode(factors)
        reconstructed_image = self.detensorize(reconstructed_tensor, size)
        return reconstructed_image

    def forward(self, x: Tensor, *args, **kwargs) -> Tensor:
        encoded_image = self.encode(x, *args, **kwargs)
        decoded_image = self.decode(encoded_image, size=x.shape[1:])
        return decoded_image
