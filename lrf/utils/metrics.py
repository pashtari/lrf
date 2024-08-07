from typing import Any
import functools
from operator import mul

import numpy as np
import torch
from skimage.metrics import structural_similarity


def mae(tensor1: torch.Tensor, tensor2: torch.Tensor) -> torch.Tensor:
    """
    Calculate the Mean Absolute Error (MAE) between two tensors.

    Args:
        tensor1 (torch.Tensor): The first input tensor.
        tensor2 (torch.Tensor): The second input tensor.

    Returns:
        torch.Tensor: The MAE value.
    """
    return torch.mean(torch.abs(tensor1 - tensor2), dim=(-3, -2, -1))


def mse(tensor1: torch.Tensor, tensor2: torch.Tensor) -> torch.Tensor:
    """
    Calculate the Mean Squared Error (MSE) between two tensors.

    Args:
        tensor1 (torch.Tensor): The first input tensor.
        tensor2 (torch.Tensor): The second input tensor.

    Returns:
        torch.Tensor: The MSE value.
    """
    return torch.mean((tensor1 - tensor2) ** 2, dim=(-3, -2, -1))


def relative_error(
    tensor1: torch.Tensor, tensor2: torch.Tensor, epsilon: float = 1e-16
) -> torch.Tensor:
    """
    Calculate the relative error between two tensors.

    Args:
        tensor1 (torch.Tensor): The first input tensor.
        tensor2 (torch.Tensor): The second input tensor.
        epsilon (float, optional): A small value to avoid division by zero (default: 1e-16).

    Returns:
        torch.Tensor: The relative error value.
    """
    numerator = torch.norm(tensor1 - tensor2, p=2, dim=(-3, -2, -1))
    denominator = torch.norm(tensor1, p=2, dim=(-3, -2, -1))
    return numerator / (denominator + epsilon)


def psnr(img1: torch.Tensor, img2: torch.Tensor, max_value: int = 255) -> torch.Tensor:
    """
    Calculate the Peak Signal-to-Noise Ratio (PSNR) between two images.

    Args:
        img1 (torch.Tensor): The first input image tensor.
        img2 (torch.Tensor): The second input image tensor.
        max_value (int, optional): The maximum possible pixel value of the images (default: 255).

    Returns:
        torch.Tensor: The PSNR value.
    """
    mse_value = mse(img1.float(), img2.float())
    psnr_value = 20 * torch.log10(max_value / torch.sqrt(mse_value))
    return psnr_value


def ssim(img1: torch.Tensor, img2: torch.Tensor) -> torch.Tensor:
    """
    Compute the Structural Similarity Index Measure (SSIM) between two images.

    Args:
        img1 (torch.Tensor): The first input image tensor of shape (C, H, W).
        img2 (torch.Tensor): The second input image tensor of shape (C, H, W).

    Returns:
        torch.Tensor: The SSIM index value.
    """
    img1_np = img1.cpu().numpy()
    img2_np = img2.cpu().numpy()
    data_range = img1_np.max() - img1_np.min()
    ssim_value = structural_similarity(
        img1_np, img2_np, channel_axis=0, data_range=data_range
    )
    return torch.tensor(ssim_value)


def get_memory_usage(obj: Any) -> int:
    """
    Calculate the memory usage of an object containing NumPy arrays or PyTorch tensors in bytes.

    Args:
        obj (Any): The data structure.

    Returns:
        int: The memory usage of the data structure in bytes.
    """
    if isinstance(obj, (list, tuple, set)):
        return sum(get_memory_usage(item) for item in obj)
    elif isinstance(obj, dict):
        return sum(get_memory_usage(item) for item in obj.values())
    elif isinstance(obj, bytes):
        return len(obj)
    elif isinstance(obj, np.ndarray):
        return obj.nbytes
    elif isinstance(obj, torch.Tensor):
        return obj.numel() * obj.element_size()
    else:
        raise ValueError(
            "Unsupported data type. Please provide an object containing NumPy arrays or PyTorch tensors."
        )


def compression_ratio(input: Any, compressed: Any) -> float:
    """
    Calculate the compression ratio between the original and compressed data.

    Args:
        input (Any): The original data structure.
        compressed (Any): The compressed data structure.

    Returns:
        float: The compression ratio.
    """
    input_memory = get_memory_usage(input)
    compressed_memory = get_memory_usage(compressed)
    return input_memory / compressed_memory


def prod(x: list[int]) -> int:
    """
    Calculate the product of elements in a list.

    Args:
        x (list[int]): A list of integers.

    Returns:
        int: The product of the list elements.
    """
    return functools.reduce(mul, x, 1)


def bits_per_pixel(size: tuple[int, int, int], compressed: object) -> float:
    """
    Calculate the bits per pixel for a compressed image.

    Args:
        size (tuple[int, int, int]): The size of the original image (C, H, W).
        compressed (object): The compressed data structure.

    Returns:
        float: The bits per pixel value.
    """
    num_pixels = prod(size)
    compressed_memory = get_memory_usage(compressed)
    return compressed_memory * 8 / num_pixels
