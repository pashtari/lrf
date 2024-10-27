from functools import reduce
from operator import mul

import torch
from torch import Tensor


def prod(x):
    return reduce(mul, x, 1)


def relative_error(tensor1: Tensor, tensor2: Tensor, eps: float = 1e-16):
    numerator = torch.norm(tensor1 - tensor2, p=2, dim=(-2, -1))
    denominator = torch.norm(tensor1, p=2, dim=(-2, -1))
    return numerator / (denominator + eps)


def safe_divide(numerator: Tensor, denominator: Tensor, eps: float = 1e-16) -> Tensor:
    """
    Safely divide numerator by denominator, avoiding division by zero,
    and preserving the sign of the denominator.
    """
    # Create a mask for denominators that are too close to zero
    small_denominator = torch.abs(denominator) < eps

    # Adjust denominators that are too small, preserving their sign
    adjusted_denominator = torch.where(
        small_denominator,
        eps * torch.sign(denominator),
        denominator,
    )

    return numerator / adjusted_denominator


def soft_thresholding(x, threshold):
    if threshold == 0:
        return x
    else:
        return torch.sign(x) * torch.relu(torch.abs(x) - threshold)
