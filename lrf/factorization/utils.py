from functools import reduce
from operator import mul

import torch
from torch import Tensor


def prod(x):
    return reduce(mul, x, 1)


def dot(x: Tensor, y: Tensor) -> Tensor:
    out = (x * y).flatten(-2).sum(-1, keepdim=True)
    return out


def relative_error(tensor1: Tensor, tensor2: Tensor, epsilon: float = 1e-16):
    numerator = torch.norm(tensor1 - tensor2, p=2, dim=(-2, -1))
    denominator = torch.norm(tensor1, p=2, dim=(-2, -1))
    return numerator / (denominator + epsilon)
