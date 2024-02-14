import math
import numpy as np

from typing import Tuple, Optional, Union
from abc import ABC, abstractmethod

import torch
from torch import nn, Tensor
import torch.nn.functional as F
import torch_dct as dct
from einops import rearrange
from skimage.metrics import structural_similarity


def zscore_normalize(tensor):
    mean = np.mean(dim=(-2, -1), keepdims=True)
    std = np.std(dim=(-2, -1), keepdims=True)
    normalized_tensor = (tensor - mean) / (std + 1e-8)
    return normalized_tensor


def minmax_normalize(x):
    min_val = torch.amin(x, dim=(-2, -1), keepdim=True)
    max_val = torch.amax(x, dim=(-2, -1), keepdim=True)
    normalized_tensor = (x - min_val) / (max_val - min_val)
    return normalized_tensor


def mae(image1, image2):
    return np.mean(np.abs(image1 - image2))


def mse(image1, image2):
    return np.mean((image1 - image2) ** 2)


def psnr(image1, image2):
    mse_value = mse(image1, image2)
    if mse_value == 0:
        return float("inf")  # If the MSE is zero, the PSNR is infinite
    max_pixel = np.max(image1)
    psnr_value = 20 * math.log10(max_pixel / math.sqrt(mse_value))
    return psnr_value


def ssim(image1, image2):
    ssim_value = structural_similarity(
        image1, image2, data_range=image2.max() - image2.min(), channel_axis=-1,
    )
    return ssim_value


def compress_interpolate(x, compression_ratio):
    """Compress the image by resizing using interpolation."""
    x = torch.tensor(x, dtype=torch.float32).permute(-1, 0, 1).unsqueeze(0)
    interpolate = Interpolate(mode="bilinear")
    y = interpolate(x, compression_ratio=compression_ratio)
    return torch.clip(y, 0, 1).squeeze(0).permute(1, 2, 0).to(torch.float64).numpy()


def compress_interpolate_low(x, compression_ratio):
    """Compress the image by resizing using interpolation."""
    x = torch.tensor(x, dtype=torch.float32).permute(-1, 0, 1).unsqueeze(0)
    interpolate = Interpolate(mode="bilinear")
    y = interpolate.compress(x, compression_ratio=compression_ratio)
    return torch.clip(y, 0, 1).squeeze(0).permute(1, 2, 0).to(torch.float64).numpy()


def compress_dct(x, compression_ratio):
    """Compress the image using DCT."""
    x = torch.tensor(x, dtype=torch.float32).permute(-1, 0, 1).unsqueeze(0)
    dct2 = DCT()
    y = dct2(x, compression_ratio=compression_ratio)
    return torch.clip(y, 0, 1).squeeze(0).permute(1, 2, 0).to(torch.float64).numpy()


def compress_dct_low(x, compression_ratio):
    """Compress the image using DCT."""
    x = torch.tensor(x, dtype=torch.float32).permute(-1, 0, 1).unsqueeze(0)
    dct2 = DCT()
    y = dct2(x, compression_ratio=compression_ratio, pad=False)
    return minmax_normalize(y).squeeze(0).permute(1, 2, 0).to(torch.float64).numpy()


def compress_svd(x, compression_ratio):
    """Compress the image using SVD."""
    x = torch.tensor(x, dtype=torch.float32).permute(-1, 0, 1).unsqueeze(0)
    svd = SVD()
    y = svd(x, compression_ratio=compression_ratio)
    return torch.clip(y, 0, 1).squeeze(0).permute(1, 2, 0).to(torch.float64).numpy()


def compress_patch_svd(x, compression_ratio, patch_size=(8, 8)):
    """Compress the image using SVD on patches."""
    x = torch.tensor(x, dtype=torch.float32).permute(-1, 0, 1).unsqueeze(0)
    patch_svd = PatchSVD(patch_size=patch_size)
    y = patch_svd(x, compression_ratio=compression_ratio)
    return torch.clip(y, 0, 1).squeeze(0).permute(1, 2, 0).to(torch.float64).numpy()

def relative_error(x: Tensor, y: Tensor, epsilon: float = 1e-16) -> Tensor:
    """Calculate the relative error between two tensors.

    Args:
        x (Tensor): The original tensor.
        y (Tensor): The tensor to compare against.
        epsilon (float): A small value to prevent division by zero.

    Returns:
        Tensor: The relative error between `x` and `y`.
    """
    numerator = torch.norm(x - y, p=2)
    denominator = torch.norm(x, p=2)
    return numerator / (denominator + epsilon)


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

    def calculate_new_size(
        self, original_size: Tuple[int, int], compression_ratio: float
    ) -> Tuple[int, int]:
        """Calculate new size for an image given a compression ratio.

        Args:
            original_size (Tuple[int, int]): The original (height, width) of the image.
            compression_ratio (float): The desired compression ratio.

        Returns:
            Tuple[int, int]: The new (height, width) of the image.
        """
        height, width = original_size
        new_height = max(math.floor(height / math.sqrt(compression_ratio)), 1)
        new_width = max(math.floor(width / math.sqrt(compression_ratio)), 1)
        return new_height, new_width

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
            new_size = self.calculate_new_size(original_size, compression_ratio)

        resized_image = F.interpolate(x, size=new_size, **self.interpolation_kwargs)

        orig_height, orig_width = original_size
        resized_height, resized_width = new_size
        self._ratio = (orig_height*orig_width)/(resized_height*resized_width)
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
        height, width = size
        cutoff_height = max(math.floor(height / (math.sqrt(compression_ratio))), 1)
        cutoff_width = max(math.floor(width / math.sqrt(compression_ratio)), 1)
        return (cutoff_height, cutoff_width)

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

        x_dct = dct.dct_2d(x)  # Perform DCT
        x_dct = x_dct[..., : cutoff[0], : cutoff[1]]  # Keep frequencies up to cutoff

        orig_height, orig_width = original_size
        resized_height, resized_width = x_dct.shape[-2:]
        self._ratio = (orig_height*orig_width)/(resized_height*resized_width)
        return x_dct

    def decompress(self, x: Tensor, size: Tuple[int, int], pad: bool = True) -> Tensor:
        """Decompress the input tensor using inverse DCT.

        Args:
            x: Tensor, the compressed tensor to be decompressed.
            size: Tuple[int, int], the original height and width of the input tensor.
            pad: bool, whether to pad the tensor to its original size.

        Returns:
            Tensor, the decompressed tensor.
        """
        height, width = size
        if pad:
            cutoff_height, cutoff_width = x.shape[-2:]
            x = F.pad(x, pad=(0, width - cutoff_width, 0, height - cutoff_height))

        y = dct.idct_2d(x)  # Perform inverse DCT
        return y

    def forward(self, x: Tensor, *args, pad=True, **kwargs) -> Tensor:
        # x: B × C × H × W
        z = self.compress(x, *args, **kwargs)
        y = self.decompress(z, x.shape[-2:], pad=pad)
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
        rows, cols = size
        df_input = rows * cols  # Degrees of freedom of the input matrix
        df_lowrank = rows + cols  # Degrees of freedom of the low-rank matrix
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
        rows, cols = size
        df_input = rows * cols  # Degrees of freedom of the input matrix
        df_lowrank = rows + cols  # Degrees of freedom of the low-rank matrix
        compression_ratio = df_input / (rank * df_lowrank)
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

        self._ratio = self.get_compression_ratio(size=original_size, rank=rank)
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
        super(PatchSVD, self).__init__()
        self.patch_size = patch_size

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
        u_new = rearrange(u, "b (h w) r -> b r 1 h w", h=x.shape[-2] // p1)
        v_new = rearrange(v, "b r (c p1 p2) -> b r c p1 p2", p1=p1, p2=p2)
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

        self._ratio = self.get_compression_ratio(size=original_size, rank=rank)
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
