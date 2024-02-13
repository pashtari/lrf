import math
from typing import Any, Tuple, Optional
from abc import ABC, abstractmethod

import torch
from torch import nn, Tensor
import torch.nn.functional as F
import torch_dct as dct
from einops import rearrange


def relative_error(x: Tensor, y: Tensor, eps: float = 1e-16) -> Tensor:
    """
    Calculates the relative error between two tensors.

    Args:
        x (Tensor): The first tensor.
        y (Tensor): The second tensor to compare against.
        eps (float, optional): A small value to avoid division by zero. Defaults to 1e-16.

    Returns:
        Tensor: The relative error between x and y.
    """
    numerator = torch.norm(x - y, p=2)
    denominator = torch.norm(x, p=2)
    return numerator / (denominator + eps)


class Compress(ABC, nn.Module):
    """Base module for compression."""

    @abstractmethod
    def compress(self, x: Tensor, *args, **kwargs) -> Any:
        pass

    @abstractmethod
    def decompress(self, *args, **kwargs) -> Any:
        pass

    def loss(self, x: Tensor, *args, **kwargs) -> Tensor:
        """
        Calculates the loss (relative error) between the original and decompressed tensors.

        Args:
            x (Tensor): The original tensor.
            u (Tensor): The left factor matrix.
            v (Tensor): The right factor matrix.

        Returns:
            Tensor: The loss value.
        """
        y = self.decompress(*args, **kwargs)
        return relative_error(x, y)

    def forward(self, x: Tensor, *args, **kwargs) -> Tensor:
        # x: B × C × H × W
        z = self.compress(x, *args, **kwargs)
        y = self.decompress(z)
        return y


class Interpolate(Compress):
    """
    A module for interpolation to compress and decompress tensors,
    allowing for dimensionality reduction and data compression.
    """

    def __init__(self, **kwargs) -> None:
        super(Interpolate, self).__init__()
        self.kwargs = kwargs

    def get_new_size(
        self, original_size: Tuple[int, int], compression_ratio: float
    ) -> int:
        """
        Calculates the new size for the interpolation based the original size and compression ratio.

        Args:
            size (Tuple[int, int]): The size of the matrix (M, N).
            compression_ratio (float): The desired compression ratio.

        Returns:
            int: The calculated new size for the compressed image.
        """
        H, W = original_size
        new_height = max(math.floor(H / math.sqrt(compression_ratio)), 1)
        new_width = max(math.floor(W / math.sqrt(compression_ratio)), 1)
        return (new_height, new_width)

    def get_compression_ratio(
        self, original_size: Tuple[int, int], new_size: Tuple[int, int]
    ) -> float:
        """
        Calculates the compression ratio based on the matrix size and cutoff.

        Args:
            size (Tuple[int, int]): The size of the original image.
            new_size (Tuple[int, int]): The size of the resized image.

        Returns:
            float: The calculated compression ratio.
        """
        H, W = original_size
        M, N = new_size
        df_original = H * W  # degrees of freedom of the original image
        df_new = M * N  # degrees of freedom of the new image
        compression_ratio = df_original / df_new
        return compression_ratio

    def compress(
        self,
        x: Tensor,
        compression_ratio: Optional[float] = None,
        new_size: Optional[Tuple[int, int]] = None,
    ) -> Tensor:
        """
        Compresses a tensor using SVD based on either a specified cutoff or compression ratio.

        Args:
            x (Tensor): The input tensor to compress.
            compression_ratio (Optional[float], optional): The desired compression ratio. Defaults to None.
            cutoff (Optional[int], optional): The specific cutoff to use for compression. Defaults to None.

        Returns:
            Tensor: The low-path filtered tensor in the frequency domain.
        """
        assert (new_size is not None) or (
            compression_ratio is not None
        ), "Either 'cutoff' or 'compression_ratio' must be specified."

        original_size = x.shape[-2:]
        if new_size is None:
            new_size = self.get_new_size(original_size, compression_ratio)

        x_resized = F.interpolate(x, size=new_size, **self.kwargs)
        return x_resized

    def decompress(self, x: Tensor, original_size: Tuple[int, int]) -> Tensor:
        """
        Decompresses a tensor from its compressed components.

        Args:
            x (Tensor): The compressed low-path filtered image in the frequency domain.
            size (Tuple[int, int]): The original spatial size of the uncompressed image.

        Returns:
            Tensor: The decompressed tensor.
        """
        y = F.interpolate(x, size=original_size, **self.kwargs)
        return y

    def forward(self, x: Tensor, *args, **kwargs) -> Tensor:
        # x: B × C × H × W
        z = self.compress(x, *args, **kwargs)
        y = self.decompress(z, x.shape[-2:])
        return y


class SVD(Compress):
    """
    A module for Singular Value Decomposition (SVD) to compress and decompress tensors,
    allowing for dimensionality reduction and data compression.
    """

    def get_rank(self, size: Tuple[int, int], compression_ratio: float) -> int:
        """
        Calculates the rank for the SVD compression based on the matrix size and compression ratio.

        Args:
            size (Tuple[int, int]): The size of the matrix (M, N).
            compression_ratio (float): The desired compression ratio.

        Returns:
            int: The calculated rank for the compressed matrix.
        """
        M, N = size
        df_input = M * N  # degrees of freedom of the input matrix
        df_lowrank = M + N  # degrees of freedom of the low-rank matrix
        rank = max(math.floor(df_input / (compression_ratio * df_lowrank)), 1)
        return rank

    def get_compression_ratio(self, size: Tuple[int, int], rank: int) -> float:
        """
        Calculates the compression ratio based on the matrix size and rank.

        Args:
            size (Tuple[int, int]): The size of the matrix (M, N).
            rank (int): The rank of the low-rank approximation.

        Returns:
            float: The calculated compression ratio.
        """
        H, W = size
        df_input = H * W  # degrees of freedom of the input matrix
        df_lowrank = H + W  # degrees of freedom of the low-rank matrix
        compression_ratio = df_input / (rank * df_lowrank)
        return compression_ratio

    def compress(
        self,
        x: Tensor,
        compression_ratio: Optional[float] = None,
        rank: Optional[int] = None,
    ) -> Tuple[Tensor, Tensor]:
        """
        Compresses a tensor using SVD based on either a specified rank or compression ratio.

        Args:
            x (Tensor): The input tensor to compress.
            compression_ratio (Optional[float], optional): The desired compression ratio. Defaults to None.
            rank (Optional[int], optional): The specific rank to use for compression. Defaults to None.

        Returns:
            Tuple[Tensor, Tensor]: The factor matrices (u, v) of the tensor.
        """
        assert (rank is not None) or (
            compression_ratio is not None
        ), "Either 'rank' or 'compression_ratio' must be specified."

        size = x.shape[-2:]
        if rank is None:
            rank = self.get_rank(size, compression_ratio)

        # Perform SVD
        u, s, vt = torch.linalg.svd(x, full_matrices=False)
        # Keep top 'rank' components
        u, s, vt = u[..., :rank], s[..., :rank], vt[..., :rank, :]

        # Distribute matrix s between u and vt
        u = torch.einsum("...ir, ...r -> ...ir", u, torch.sqrt(s))
        v = torch.einsum("...r, ...rj -> ...jr", torch.sqrt(s), vt)
        return u, v

    def decompress(self, x: Tuple[Tensor, Tensor]) -> Tensor:
        """
        Decompresses a tensor from its compressed components.

        Args:
            x (Tuple[Tensor, Tensor]): The factor matrices (u, v) of the tensor.

        Returns:
            Tensor: The decompressed tensor.
        """
        u, v = x
        y = u @ v.transpose(-2, -1)
        return y


class PatchSVD(SVD):
    """
    A module for Singular Value Decomposition (SVD) on patches to compress and decompress tensors,
    allowing for dimensionality reduction and data compression.
    """

    def __init__(self, patch_size=(8, 8)) -> None:
        super(PatchSVD, self).__init__()
        self.patch_size = patch_size

    def patchify(self, x):
        p1, p2 = self.patch_size
        patches = rearrange(x, "b c (h p1) (w p2) -> b (h w) (c p1 p2)", p1=p1, p2=p2)
        return patches

    def depatchify_uv(self, x, u, v):
        p1, p2 = self.patch_size
        u_new = rearrange(u, "b (h w) r -> b r 1 h w", h=x.shape[-2] // p1)
        v_new = rearrange(v, "b r (c p1 p2) -> b r c p1 p2", p1=p1, p2=p2)
        return u_new, v_new

    def depatchify(self, x, size):
        p1, p2 = self.patch_size
        H, _ = size
        patches = rearrange(
            x, "b (h w) (c p1 p2) -> b c (h p1) (w p2)", p1=p1, p2=p2, h=H // p1
        )
        return patches

    def compress(
        self,
        x: Tensor,
        compression_ratio: Optional[float] = None,
        rank: Optional[int] = None,
    ) -> Tuple[Tensor, Tensor]:

        assert (rank is not None) or (
            compression_ratio is not None
        ), "Either 'rank' or 'compression_ratio' must be specified."

        # Patchify
        patches = self.patchify(x)

        # Set rank or compression ratio
        size = patches.shape[-2:]
        if rank is None:
            rank = self.get_rank(size, compression_ratio)

        # Perform SVD
        u, s, vt = torch.linalg.svd(patches, full_matrices=False)
        # Keep top 'rank' components
        u, s, vt = u[..., :rank], s[..., :rank], vt[..., :rank, :]

        # Distribute matrix s between u and vt
        u = torch.einsum("...ir, ...r -> ...ir", u, torch.sqrt(s))
        v = torch.einsum("...r, ...rj -> ...jr", torch.sqrt(s), vt)
        # u = u[:, :, :rank] @ torch.diag_embed(torch.sqrt(s[:, :rank]))
        # v = vt[:, :rank, :].transpose(-2, -1) @ torch.diag_embed(torch.sqrt(s[:, :rank]))
        return u, v

    def decompress(self, x: Tuple[Tensor, Tensor], size: Tuple[int, int]) -> Tensor:
        u, v = x
        reconstructed_patches = u @ v.transpose(-2, -1)
        reconstructed_image = self.depatchify(reconstructed_patches, size)
        return reconstructed_image

    def forward(self, x: Tensor, *args, **kwargs) -> Tensor:
        # x: B × C × H × W
        z = self.compress(x, *args, **kwargs)
        y = self.decompress(z, x.shape[-2:])
        return y


class DCT(Compress):
    """
    A module for Discrete Cosine Transform (DCT) to compress and decompress tensors,
    allowing for dimensionality reduction and data compression.
    """

    def get_cutoff(
        self, size: Tuple[int, int], compression_ratio: float
    ) -> Tuple[int, int]:
        """
        Calculates the cutoff for the DCT-compression based the matrix size and compression ratio.

        Args:
            size (Tuple[int, int]): The size of the matrix (M, N).
            compression_ratio (float): The desired compression ratio.

        Returns:
            int: The calculated cutoff for the compressed matrix.
        """
        H, W = size
        cutoff_h = max(math.floor(H / (math.sqrt(compression_ratio))), 1)
        cutoff_w = max(math.floor(W / math.sqrt(compression_ratio)), 1)
        return (cutoff_h, cutoff_w)

    def get_compression_ratio(
        self, size: Tuple[int, int], cutoff: Tuple[int, int]
    ) -> float:
        """
        Calculates the compression ratio based on the matrix size and cutoff.

        Args:
            size (Tuple[int, int]): The size of the matrix (M, N).
            cutoff (int): The cutoff of the low-path filter.

        Returns:
            float: The calculated compression ratio.
        """
        M, N = size
        cutoff_h, cutoff_w = cutoff
        df_input = M * N  # degrees of freedom of the input matrix
        compression_ratio = df_input / (cutoff_h * cutoff_w)
        return compression_ratio

    def compress(
        self,
        x: Tensor,
        compression_ratio: Optional[float] = None,
        cutoff: Optional[Tuple[int, int]] = None,
    ) -> Tuple[Tensor, Tensor]:
        """
        Compresses a tensor using SVD based on either a specified cutoff or compression ratio.

        Args:
            x (Tensor): The input tensor to compress.
            compression_ratio (Optional[float], optional): The desired compression ratio. Defaults to None.
            cutoff (Optional[int], optional): The specific cutoff to use for compression. Defaults to None.

        Returns:
            Tensor: The low-path filtered tensor in the frequency domain.
        """
        assert (cutoff is not None) or (
            compression_ratio is not None
        ), "Either 'cutoff' or 'compression_ratio' must be specified."

        size = x.shape[-2:]
        if cutoff is None:
            cutoff = self.get_cutoff(size, compression_ratio)

        # Take DCT
        x_dct = dct.dct_2d(x)

        # Keep up to cutoff frequency
        x_dct = x_dct[..., : cutoff[0], : cutoff[1]]
        return x_dct

    def decompress(self, x: Tensor, size: Tuple[int, int], pad=True) -> Tensor:
        """
        Decompresses a tensor from its compressed components.

        Args:
            x (Tensor): The compressed low-path filtered image in the frequency domain.
            size (Tuple[int, int]): The original spatial size of the uncompressed image.

        Returns:
            Tensor: The decompressed tensor.
        """
        H, W = size
        if pad:
            cutoff_h, cutoff_w = x.shape[-2:]
            x = F.pad(x, pad=(0, H - cutoff_h, 0, W - cutoff_w))

        y = dct.idct_2d(x)
        return y

    def forward(self, x: Tensor, *args, pad=True, **kwargs) -> Tensor:
        # x: B × C × H × W
        z = self.compress(x, *args, **kwargs)
        y = self.decompress(z, x.shape[-2:], pad=pad)
        return y
