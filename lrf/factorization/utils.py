from functools import reduce
from operator import mul

import torch
from torch import Tensor


def prod(x):
    return reduce(mul, x, 1)


def relative_error(tensor1: Tensor, tensor2: Tensor, epsilon: float = 1e-16):
    numerator = torch.norm(tensor1 - tensor2, p=2, dim=(-2, -1))
    denominator = torch.norm(tensor1, p=2, dim=(-2, -1))
    return numerator / (denominator + epsilon)


def soft_thresholding(x, threshold):
    if threshold == 0:
        return x
    else:
        return torch.sign(x) * torch.relu(torch.abs(x) - threshold)
