from functools import reduce
from operator import mul

import torch
import numpy as np


def prod(x):
    return reduce(mul, x, 1)


def zscore_normalize(tensor, dim=(-2, -1), eps=1e-8):
    mean = torch.mean(tensor, dim=dim, keepdim=True)
    std = torch.std(tensor, dim=dim, keepdim=True)
    normalized_tensor = (tensor - mean) / (std + eps)
    return normalized_tensor


def minmax_normalize(tensor, dim=(-2, -1), eps=1e-8):
    min_val = torch.amin(tensor, dim=dim, keepdim=True)
    max_val = torch.amax(tensor, dim=dim, keepdim=True)
    normalized_tensor = (tensor - min_val) / (max_val - min_val + eps)
    return normalized_tensor


def tensor_memory_usage(data):
    """
    Calculates the memory usage of a NumPy array or a PyTorch tensor in bytes.

    Args:
    - data (numpy.ndarray or torch.Tensor): The data structure.

    Returns:
    - int: The memory usage of the data structure in bytes.
    """
    if isinstance(data, np.ndarray):
        # For NumPy array, use the nbytes attribute and convert to bits
        return data.nbytes
    elif isinstance(data, torch.Tensor):
        # For PyTorch tensor, calculate as before
        return data.numel() * data.element_size()
    else:
        raise ValueError(
            "Unsupported data type. Please provide a NumPy array or a PyTorch tensor."
        )
