from typing import Tuple, Optional, Union
from abc import ABC, abstractmethod
import math

import torch
from torch import nn, Tensor
import torch.nn.functional as F
from torch.nn.modules.utils import _pair
import torch_dct as dct
from einops import rearrange

from .metrics import relative_error
from .utils import zscore_normalize


class Compress(ABC, nn.Module):
    """Abstract base class for compression methods."""

    @abstractmethod
    def compress(self, x: Tensor, *args, **kwargs) -> Union[Tensor, Tuple[Tensor, ...]]:
        """Compress the input tensor."""
        pass

    @abstractmethod
    def decompress(self, *args, **kwargs) -> Tensor:
        """Decompress the input tensor."""
        pass

    def loss(self, x: Tensor, *args, **kwargs) -> Tensor:
        """Calculate the loss between the original and decompressed tensors.

        Args:
            x (Tensor): The original tensor.

        Returns:
            Tensor: The calculated loss.
        """
        y = self.decompress(*args, **kwargs)
        return relative_error(x, y)

    def forward(self, x: Tensor, *args, **kwargs) -> Tensor:
        """Forward pass through the compression model.

        Args:
            x (Tensor): Input tensor to be compressed and decompressed.

        Returns:
            Tensor: The decompressed tensor.
        """
        compressed = self.compress(x, *args, **kwargs)
        decompressed = self.decompress(compressed)
        return decompressed


class Interpolate(Compress):
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

    def compress(
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

    def decompress(self, x: Tensor, original_size: Tuple[int, int]) -> Tensor:
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
        z = self.compress(x, *args, **kwargs)
        y = self.decompress(z, x.shape[-2:])
        return y


class DCT(Compress):
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

    def compress(
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

    def decompress(
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
        z = self.compress(x, *args, **kwargs)
        y = self.decompress(z, x.shape[-2:], pad=pad, zscore=zscore)
        return y


class SVD(Compress):
    """Implements Singular Value Decomposition (SVD) based compression."""

    def get_rank(self, size: Tuple[int, int], compression_ratio: float) -> int:
        """Calculate the rank for SVD based on the compression ratio.

        Args:
            size (Tuple[int, int]): The size of the input tensor.
            compression_ratio (float): The desired compression ratio.

        Returns:
            int: The calculated rank.
        """
        size = _pair(size)
        num_rows, num_cols = size
        df_input = num_rows * num_cols  # Degrees of freedom of the input matrix
        df_lowrank = num_rows + num_cols  # Degrees of freedom of the low-rank matrix
        rank = max(math.floor(df_input / (compression_ratio * df_lowrank)), 1)
        return rank

    def get_compression_ratio(self, size: Tuple[int, int], rank: int) -> float:
        """Calculate the compression ratio based on the rank.

        Args:
            size (Tuple[int, int]): The size of the input tensor.
            rank (int): The rank used for compression.

        Returns:
            float: The compression ratio.
        """
        size = _pair(size)
        num_rows, num_cols = size
        df_input = num_rows * num_cols  # Degrees of freedom of the input matrix
        df_lowrank = rank * (
            num_rows + num_cols
        )  # Degrees of freedom of the low-rank matrix
        compression_ratio = df_input / df_lowrank
        return compression_ratio

    def compress(
        self,
        x: Tensor,
        compression_ratio: Optional[float] = None,
        rank: Optional[int] = None,
    ) -> Tuple[Tensor, Tensor]:
        """Compress the input tensor using SVD.

        Args:
            x (Tensor): The input tensor to compress.
            compression_ratio (Optional[float], optional): The desired compression ratio. Defaults to None.
            rank (Optional[int], optional): The rank to use for SVD. Defaults to None.

        Returns:
            Tuple[Tensor, Tensor]: The compressed form of the input tensor.
        """
        assert (rank is not None) or (
            compression_ratio is not None
        ), "Either 'rank' or 'compression_ratio' must be specified."

        original_size = x.shape[-2:]
        if rank is None:
            rank = self.get_rank(original_size, compression_ratio)

        u, s, vt = torch.linalg.svd(x, full_matrices=False)
        u, s, vt = u[..., :rank], s[..., :rank], vt[..., :rank, :]

        u = torch.einsum("...ir, ...r -> ...ir", u, torch.sqrt(s))
        v = torch.einsum("...r, ...rj -> ...jr", torch.sqrt(s), vt)

        self.real_compression_ratio = self.get_compression_ratio(
            size=original_size, rank=rank
        )
        return u, v

    def decompress(self, x: Tuple[Tensor, Tensor]) -> Tensor:
        """Decompress the input tensor from its compressed form.

        Args:
            x (Tuple[Tensor, Tensor]): The compressed form of the input tensor.

        Returns:
            Tensor: The decompressed tensor.
        """
        u, v = x
        y = u @ v.transpose(-2, -1)
        return y


class PatchSVD(SVD):
    """Implements SVD-based compression with patchification."""

    def __init__(self, patch_size: Tuple[int, int] = (8, 8)) -> None:
        """Initialize the PatchSVD compressor.

        Args:
            patch_size (Tuple[int, int], optional): The size of the patches. Defaults to (8, 8).
        """
        super().__init__()
        self.patch_size = _pair(patch_size)

    def patchify(self, x: Tensor) -> Tensor:
        """Break down the input tensor into patches.

        Args:
            x (Tensor): The input tensor.

        Returns:
            Tensor: The tensor reshaped into patches.
        """
        p1, p2 = self.patch_size
        patches = rearrange(x, "b c (h p1) (w p2) -> b (h w) (c p1 p2)", p1=p1, p2=p2)
        return patches

    def depatchify_uv(self, x: Tensor, u: Tensor, v: Tensor) -> Tuple[Tensor, Tensor]:
        """Reshape the u and v matrices into their original spatial dimensions.

        Args:
            x (Tensor): The original input tensor.
            u (Tensor): The u matrix from SVD.
            v (Tensor): The v matrix from SVD.

        Returns:
            Tuple[Tensor, Tensor]: The u and v matrices reshaped.
        """
        p1, p2 = self.patch_size
        u_new = rearrange(u, "b (h w) r -> b 1 r h w", h=x.shape[-2] // p1)
        v_new = rearrange(v, "b (c p1 p2) r -> b c r p1 p2", p1=p1, p2=p2)
        return u_new, v_new

    def depatchify(self, x: Tensor, size: Tuple[int, int]) -> Tensor:
        """Reconstruct the original tensor from its patches.

        Args:
            x (Tensor): The tensor in its patch form.
            size (Tuple[int, int]): The size of the original tensor.

        Returns:
            Tensor: The reconstructed tensor.
        """
        p1, p2 = self.patch_size
        patches = rearrange(
            x, "b (h w) (c p1 p2) -> b c (h p1) (w p2)", p1=p1, p2=p2, h=size[0] // p1
        )
        return patches

    def compress(
        self,
        x: Tensor,
        compression_ratio: Optional[float] = None,
        rank: Optional[int] = None,
    ) -> Tuple[Tensor, Tensor]:
        """Compress the input tensor using patch-wise SVD.

        Overrides the compress method from the SVD class to apply patch-wise compression.

        Args:
            x (Tensor): The input tensor to compress.
            compression_ratio (Optional[float], optional): The desired compression ratio. Defaults to None.
            rank (Optional[int], optional): The rank to use for SVD. Defaults to None.

        Returns:
            Tuple[Tensor, Tensor]: The compressed form of the input tensor.
        """
        patches = self.patchify(x)

        original_size = patches.shape[-2:]
        if rank is None:
            rank = self.get_rank(original_size, compression_ratio)

        u, s, vt = torch.linalg.svd(patches, full_matrices=False)
        u, s, vt = u[..., :rank], s[..., :rank], vt[..., :rank, :]

        u = torch.einsum("...ir, ...r -> ...ir", u, torch.sqrt(s))
        v = torch.einsum("...r, ...rj -> ...jr", torch.sqrt(s), vt)

        self.real_compression_ratio = self.get_compression_ratio(
            size=original_size, rank=rank
        )
        return u, v

    def decompress(self, x: Tuple[Tensor, Tensor], size: Tuple[int, int]) -> Tensor:
        """Decompress the input tensor from its compressed patch form.

        Args:
            x (Tuple[Tensor, Tensor]): The compressed form of the input tensor.
            size (Tuple[int, int]): The size of the original tensor.

        Returns:
            Tensor: The decompressed tensor.
        """
        u, v = x
        reconstructed_patches = u @ v.transpose(-2, -1)
        reconstructed_image = self.depatchify(reconstructed_patches, size)
        return reconstructed_image

    def forward(self, x: Tensor, *args, **kwargs) -> Tensor:
        """Defines the computation performed at every call for patch-wise compression.

        Args:
            x (Tensor): The input tensor.

        Returns:
            Tensor: The decompressed tensor, after compression and decompression.
        """
        compressed_patches = self.compress(x, *args, **kwargs)
        decompressed_image = self.decompress(compressed_patches, x.shape[-2:])
        return decompressed_image


def filter_topk_elements(x, k):
    # Calculate the absolute values of X
    absolute_x = torch.abs(x)

    # Flatten the tensor to work with absolute values
    flattened_x = torch.flatten(absolute_x, -2, -1)

    # Use kthvalue to find the k largest value and it index
    kth_value = torch.kthvalue(flattened_x, flattened_x.shape[-1] - k + 1).values

    kth_value_unsqueezed = kth_value.unsqueeze(-1).unsqueeze(-1)

    # Use torch.where to create a mask and apply it directly
    y = torch.where(absolute_x >= kth_value_unsqueezed, x, 0.0)
    return y


class LSD(Compress):
    """Implements Low-Rank Plus Sparse Decomposition (LSD) based compression."""

    def __init__(self, num_iters: int = 100, verbose: bool = False) -> None:
        super().__init__()
        self.num_iters = num_iters
        self.verbose = verbose

    def get_rank(
        self, size: Tuple[int, int], compression_ratio: float, alpha: float
    ) -> int:
        size = _pair(size)
        num_rows, num_cols = size
        numerator = num_rows * num_cols * (1 - 2 * alpha * compression_ratio)
        denominator = (num_rows + num_cols) * compression_ratio
        rank = max(math.floor(numerator / denominator), 1)
        return rank

    def get_alpha(
        self, size: Tuple[int, int], compression_ratio: float, rank: int
    ) -> int:
        size = _pair(size)
        num_rows, num_cols = size
        numel = num_rows * num_cols
        numerator = numel - compression_ratio * rank * (num_rows + num_cols)
        denominator = 2 * compression_ratio * numel
        alpha = max(numerator / denominator, 0)
        alpha = math.floor(alpha * numel) / numel
        return alpha

    def get_compression_ratio(
        self, size: Tuple[int, int], rank: int, alpha: float
    ) -> float:
        size = _pair(size)
        num_rows, num_cols = size
        df_input = num_rows * num_cols  # Degrees of freedom of the input matrix
        df_lowrank = rank * (
            num_rows + num_cols
        )  # Degrees of freedom of the low-rank matrix
        df_sparse = (
            math.floor(num_rows * num_cols * alpha) * 2
        )  # Degrees of freedom of the sparse matrix
        compression_ratio = df_input / (df_lowrank + df_sparse)
        return compression_ratio

    def update_uv(self, x: Tensor, s: Optional[Tensor], rank: int):
        e = x - s if isinstance(s, Tensor) else x
        u, s, vt = torch.linalg.svd(e, full_matrices=False)
        u, s, vt = u[..., :rank], s[..., :rank], vt[..., :rank, :]

        u = torch.einsum("...ir, ...r -> ...ir", u, torch.sqrt(s))
        v = torch.einsum("...r, ...rj -> ...jr", torch.sqrt(s), vt)
        return u, v

    def update_s(self, x: Tensor, u: Tensor, v: Tensor, alpha: float):
        if alpha == 0:
            s = torch.zeros_like(x)
        else:
            e = x - u @ v.transpose(-2, -1)
            num_rows, num_cols = x.shape[-2:]
            zero_norm = math.floor(alpha * num_rows * num_cols)
            if zero_norm == 0:
                s = torch.zeros_like(x)
            else:
                s = filter_topk_elements(e, zero_norm)
            # update formula for l1 regularization
            # lam = (1 - alpha) / alpha
            # s = torch.relu(torch.abs(e) - lam / 2)
        return s

    def compress(
        self, x: Tensor, alpha: float, rank: int = None, compression_ratio: float = None
    ) -> Tuple[Tensor, Tensor, Tensor]:
        # x: B × * × M × N
        size = x.shape[-2:]

        # init
        u, v, s = None, None, None

        if rank is None:
            assert (
                compression_ratio is not None
            ), "at least rank or compression ration should be provided"
            rank = self.get_rank(size, compression_ratio, alpha)

        # iterate
        for it in range(1, self.num_iters + 1):
            u, v = self.update_uv(x, s, rank)

            s = self.update_s(x, u, v, alpha)

            if self.verbose:
                loss = self.loss(x, (u, v, s))
                print(f"Iter {it}, loss = {loss}")

        self.real_compression_ratio = self.get_compression_ratio(size, rank, alpha)
        return u, v, s

    def decompress(self, x: Tuple[Tensor, Tensor, Tensor]) -> Tensor:
        u, v, s = x
        y = u @ v.transpose(-2, -1) + s
        return y


class PatchLSD(LSD):
    """Implements Patch Low-Rank Plus Sparse Decomposition (PatchLSD) based compression."""

    def __init__(
        self,
        patch_size: Tuple[int, int] = (8, 8),
        num_iters: int = 100,
        verbose: bool = False,
    ) -> None:
        super().__init__()
        self.num_iters = num_iters
        self.verbose = verbose
        self.patch_size = _pair(patch_size)

    def patchify(self, x: Tensor) -> Tensor:
        p1, p2 = self.patch_size
        patches = rearrange(x, "b c (h p1) (w p2) -> b (h w) (c p1 p2)", p1=p1, p2=p2)
        return patches

    def depatchify_uv(self, x: Tensor, u: Tensor, v: Tensor) -> Tuple[Tensor, Tensor]:
        p1, p2 = self.patch_size
        u_new = rearrange(u, "b (h w) r -> b 1 r h w", h=x.shape[-2] // p1)
        v_new = rearrange(v, "b (c p1 p2) r -> b c r p1 p2", p1=p1, p2=p2)
        return u_new, v_new

    def depatchify(self, x: Tensor, size: Tuple[int, int]) -> Tensor:
        p1, p2 = self.patch_size
        patches = rearrange(
            x, "b (h w) (c p1 p2) -> b c (h p1) (w p2)", p1=p1, p2=p2, h=size[0] // p1
        )
        return patches

    def compress(
        self, x: Tensor, alpha: float, rank: int = None, compression_ratio: float = None
    ) -> Tuple[Tensor, Tensor, Tensor]:
        # x: B × * × M × N

        patches = self.patchify(x)

        if rank is None:
            assert (
                compression_ratio is not None
            ), "at least rank or compression ration should be provided"
            size = patches.shape[-2:]
            rank = self.get_rank(size, compression_ratio, alpha)

        # init
        u, v, s = None, None, None

        # iterate
        for it in range(1, self.num_iters + 1):
            u, v = self.update_uv(patches, s, rank)

            s = self.update_s(patches, u, v, alpha)

            if self.verbose:
                loss = self.loss(x, (u, v, s), x.shape[-2:])
                print(f"Iter {it}, loss = {loss}")

        self.real_compression_ratio = self.get_compression_ratio(
            patches.shape[-2:], rank, alpha
        )
        return u, v, s

    def decompress(
        self, x: Tuple[Tensor, Tensor, Tensor], size: Tuple[int, int]
    ) -> Tensor:
        u, v, s = x
        reconstructed_patches = u @ v.transpose(-2, -1) + s
        reconstructed_image = self.depatchify(reconstructed_patches, size)
        return reconstructed_image

    def forward(self, x: Tensor, *args, **kwargs) -> Tensor:
        compressed_patches = self.compress(x, *args, **kwargs)
        decompressed_image = self.decompress(compressed_patches, x.shape[-2:])
        return decompressed_image


# class ALSD(Compress):
#     """Implements Low-Rank Plus Sparse Decomposition (LSD) based compression."""

#     def __init__(self, num_iters: int = 100, verbose: bool = False) -> None:
#         super().__init__()
#         self.num_iters = num_iters
#         self.verbose = verbose

#     def get_compression_ratio(
#         self, size: Tuple[int, int], rank: int, alpha: float
#     ) -> float:
#         size = _pair(size)
#         num_rows, num_cols = size
#         df_input = num_rows * num_cols  # Degrees of freedom of the input matrix
#         df_lowrank = rank * (
#             num_rows + num_cols
#         )  # Degrees of freedom of the low-rank matrix
#         df_sparse = (
#             num_rows * num_cols * alpha * 2
#         )  # Degrees of freedom of the sparse matrix
#         compression_ratio = df_input / (df_lowrank + df_sparse)
#         return compression_ratio

#     def get_rank(
#         self, size: Tuple[int, int], compression_ratio: float, alpha: float
#     ) -> int:
#         size = _pair(size)
#         num_rows, num_cols = size
#         numerator = num_rows * num_cols * (1 - 2 * alpha * compression_ratio)
#         denominator = (num_rows + num_cols) * compression_ratio
#         rank = max(math.floor(numerator / denominator), 1)
#         return rank

#     def get_alpha(
#         self, size: Tuple[int, int], compression_ratio: float, rank: int
#     ) -> int:
#         size = _pair(size)
#         num_rows, num_cols = size
#         numerator = num_rows * num_cols - compression_ratio * rank * (num_rows + num_cols)
#         denominator = 2 * compression_ratio * num_rows * num_cols
#         alpha = max(numerator / denominator, 0)
#         return alpha

#     def update_uv(self, x: Tensor, s: Optional[Tensor], rank: int):
#         e = x - s if isinstance(s, Tensor) else x
#         u, s, vt = torch.linalg.svd(e, full_matrices=False)
#         u, s, vt = u[..., :rank], s[..., :rank], vt[..., :rank, :]

#         u = torch.einsum("...ir, ...r -> ...ir", u, torch.sqrt(s))
#         v = torch.einsum("...r, ...rj -> ...jr", torch.sqrt(s), vt)
#         return u, v

#     def update_s(self, x: Tensor, u: Tensor, v: Tensor, alpha: float):
#         if alpha == 0.0:
#             s = torch.zeros_like(x)
#         else:
#             e = x - u @ v.transpose(-2, -1)
#             num_rows, num_cols = x.shape[-2:]
#             zero_norm = math.floor(alpha * (num_rows * num_cols))
#             s = filter_topk_elements(e, zero_norm)
#             # lam = (1 - alpha) / alpha
#             # s = torch.relu(torch.abs(e) - lam / 2)
#         return s

#     def compress(
#         self,
#         x: Tensor,
#         compression_ratio: float,
#     ) -> Tuple[Tensor, Tensor, Tensor]:
#         # x: B × * × M × N
#         size = x.shape[-2:]

#         # init
#         u, v, s = None, None, None

#         alpha = 0.0
#         rank = self.get_rank(size, compression_ratio, alpha)

#         # iterate
#         for it in range(1, self.num_iters + 1):
#             rank_values = [*range(1, rank + 1)]
#             alpha_values = []
#             loss_values = []
#             uu = []
#             vv = []
#             ss = []
#             for r in rank_values:
#                 alpha = self.get_alpha(size, compression_ratio, r)
#                 alpha_values.append(alpha)

#                 u, v = self.update_uv(x, s, r)
#                 s = self.update_s(x, u, v, alpha)

#                 uu.append(u)
#                 vv.append(v)
#                 ss.append(s)
#                 loss_values.append(self.loss(x, (u, v, s)))

#             j = torch.stack(loss_values).flatten(1).mean(1).argmin(0).item()
#             rank, alpha = rank_values[j], alpha_values[j]
#             loss = loss_values[j]
#             u, v, s = uu[j], vv[j], ss[j]

#             if self.verbose:
#                 loss = self.loss(x, (u, v, s))
#                 print(f"Iter {it}, loss = {loss}")

#         self.real_compression_ratio = self.get_compression_ratio(size, rank, alpha)
#         return u, v, s

#     def decompress(self, x: Tuple[Tensor, Tensor, Tensor]) -> Tensor:
#         u, v, s = x
#         y = u @ v.transpose(-2, -1) + s
#         return y


# class PatchALSD(ALSD):
#     """Implements Patch Low-Rank Plus Sparse Decomposition (PatchLSD) based compression."""

#     def __init__(
#         self,
#         patch_size: Tuple[int, int] = (8, 8),
#         num_iters: int = 100,
#         verbose: bool = False,
#     ) -> None:
#         super().__init__()
#         self.num_iters = num_iters
#         self.verbose = verbose
#         self.patch_size = _pair(patch_size)

#     def patchify(self, x: Tensor) -> Tensor:
#         p1, p2 = self.patch_size
#         patches = rearrange(x, "b c (h p1) (w p2) -> b (h w) (c p1 p2)", p1=p1, p2=p2)
#         return patches

#     def depatchify_uv(self, x: Tensor, u: Tensor, v: Tensor) -> Tuple[Tensor, Tensor]:
#         p1, p2 = self.patch_size
#         u_new = rearrange(u, "b (h w) r -> b 1 r h w", h=x.shape[-2] // p1)
#         v_new = rearrange(v, "b (c p1 p2) r -> b c r p1 p2", p1=p1, p2=p2)
#         return u_new, v_new

#     def depatchify(self, x: Tensor, size: Tuple[int, int]) -> Tensor:
#         p1, p2 = self.patch_size
#         patches = rearrange(
#             x, "b (h w) (c p1 p2) -> b c (h p1) (w p2)", p1=p1, p2=p2, h=size[0] // p1
#         )
#         return patches

#     def compress(
#         self,
#         x: Tensor,
#         compression_ratio: float,
#         alpha: float,
#     ) -> Tuple[Tensor, Tensor, Tensor]:
#         # x: B × * × M × N

#         patches = self.patchify(x)

#         size = patches.shape[-2:]

#         # init
#         u, v, s = None, None, None

#         rank = self.get_rank(size, compression_ratio, alpha)

#         # iterate
#         for it in range(1, self.num_iters + 1):
#             u, v = self.update_uv(patches, s, rank)

#             s = self.update_s(patches, u, v, alpha)

#             if self.verbose:
#                 loss = self.loss(x, (u, v, s), x.shape[-2:])
#                 print(f"Iter {it}, loss = {loss}")

#         self.real_compression_ratio = self.get_compression_ratio(
#             patches.shape[-2:], rank, alpha
#         )
#         return u, v, s

#     def decompress(
#         self, x: Tuple[Tensor, Tensor, Tensor], size: Tuple[int, int]
#     ) -> Tensor:
#         u, v, s = x
#         reconstructed_patches = u @ v.transpose(-2, -1) + s
#         reconstructed_image = self.depatchify(reconstructed_patches, size)
#         return reconstructed_image

#     def forward(self, x: Tensor, *args, **kwargs) -> Tensor:
#         compressed_patches = self.compress(x, *args, **kwargs)
#         decompressed_image = self.decompress(compressed_patches, x.shape[-2:])
#         return decompressed_image
