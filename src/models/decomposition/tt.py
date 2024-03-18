from typing import Sequence

import torch
from torch import Tensor
from torch.nn.modules.utils import _ntuple
import opt_einsum as oe

from .utils import prod


def tt_rank_upper_bounds(size: Sequence[int]):
    """
    Calculate the theoretical upper bounds of ranks for a tensor of arbitrary order
    in Tensor Train (TT) decomposition.

    Args:
    - size (list of int): Size of the tensor.

    Returns:
    - list of int: Upper bounds for ranks between each pair of consecutive modes.
    """
    ndim = len(size)  # Order of the tensor
    upper_bounds = []

    for i in range(1, ndim):
        left_dims_product = 1
        right_dims_product = 1

        # Product of dimensions on the left side of the split
        for j in range(i):
            left_dims_product *= size[j]

        # Product of dimensions on the right side of the split
        for j in range(i, ndim):
            right_dims_product *= size[j]

        # Upper bound is the minimum of the products on either side of the split
        upper_bound = min(left_dims_product, right_dims_product)
        upper_bounds.append(upper_bound)

    return upper_bounds


def tt_rank_feasible_ranges(size, compression_ratio):
    """
    Calculate feasible ranges of ranks for a tensor of arbitrary dimensions
    given a desired compression ratio.

    The function estimates lower and upper bounds for the ranks in a tensor
    train (TT) decomposition that would allow achieving a specified compression
    ratio.

    Args:
        size (list of int): Size of the tensor.
        compression_ratio (float): Desired compression ratio.

    Returns:
        list of tuples: Each tuple contains the lower and upper bounds
                        for a rank between consecutive modes of the tensor. The
                        bounds are estimated such that, if achievable, the TT
                        decomposition would not exceed the target storage size
                        implied by the specified compression ratio.
    """
    # Recalculate the upper bounds to ensure accuracy
    upper_bounds = tt_rank_upper_bounds(size)
    upper_bounds = [1, *upper_bounds, 1]

    # Calculate the total number of elements in the tensor
    total_elements = prod(size)

    # Target storage size based on the compression ratio
    target_storage = total_elements / compression_ratio

    feasible_ranges = []
    for i in range(1, len(upper_bounds) - 1):
        # Lower bound estimation
        # Assume all other ranks at their maximum (upper bound) for the most conservative estimate
        storage_with_max_others = sum(
            [
                upper_bounds[j] * size[j] * upper_bounds[j + 1]
                for j in range(len(size))
                if j not in (i - 1, i)
            ]
        )
        max_storage_current = target_storage - storage_with_max_others
        lower_bound = max(
            1,
            int(
                max_storage_current
                / (upper_bounds[i - 1] * size[i - 1] + upper_bounds[i + 1] * size[i])
            ),
        )

        # Upper bound estimation
        # Assume all other ranks at minimum (1) for the least conservative estimate
        storage_with_min_others = sum(
            [
                1 * size[j - 1] * 1
                for j in range(1, len(upper_bounds))
                if j not in (i, i + 1)
            ]
        )
        min_storage_current = target_storage - storage_with_min_others
        upper_bound = min(
            upper_bounds[i], int(min_storage_current / (1 * size[i - 1] + 1 * size[i]))
        )

        # Ensure logical ordering of bounds
        feasible_ranges.append((lower_bound, upper_bound))

    return feasible_ranges


def ttd(x: Tensor, rank: Sequence[int]):
    "Tensor Train (TT) decomposition."
    ranks = _ntuple(x.ndim - 1)(rank)
    assert len(ranks) == x.ndim - 1

    ranks = [float("inf") if r is None else r for r in ranks]
    ranks = [1, *ranks, 1]

    unfolding = x
    factors = [None] * x.ndim

    for k in range(x.ndim - 1):
        # Reshape the unfolding matrix of the remaining factors
        num_rows = int(ranks[k] * x.shape[k])
        unfolding = torch.reshape(unfolding, (num_rows, -1))

        # SVD of the unfolding matrix
        num_rows, num_cols = unfolding.shape
        current_rank = min(num_rows, num_cols, ranks[k + 1])
        U, S, Vt = torch.linalg.svd(unfolding, full_matrices=False)
        U, S, Vt = U[..., :current_rank], S[..., :current_rank], Vt[..., :current_rank, :]

        ranks[k + 1] = current_rank

        # Get the kth factor
        factors[k] = torch.reshape(U, (ranks[k], x.shape[k], ranks[k + 1]))

        # Get new unfolding matrix for the remaining factors
        unfolding = torch.reshape(S, (-1, 1)) * Vt

    # Getting the last factor
    prev_rank, last_dim = unfolding.shape
    factors[-1] = torch.reshape(unfolding, (prev_rank, last_dim))

    # Removing the first dimension of the first factor
    factors[0].squeeze_(0)

    return factors


batched_ttd = torch.vmap(ttd, in_dims=(0,))


def contract_tt(factors: Sequence[Tensor]):
    num_factors = len(factors)
    args = []
    # Add the first factor
    args.append(factors[0])
    args.append(["i0", "r1"])
    for d in range(1, num_factors - 1):
        args.append(factors[d])
        args.append([f"r{d}", f"i{d}", f"r{d+1}"])

    # Add the last factor
    args.append(factors[-1])
    args.append([f"r{num_factors - 1}", f"i{num_factors - 1}"])

    # Add the last factor
    args.append([f"i{d}" for d in range(num_factors)])
    return oe.contract(*args)


def batched_contract_tt(factors: Sequence[Tensor]):
    num_factors = len(factors)
    args = []
    # Add the first factor
    args.append(factors[0])
    args.append(["b", "i0", "r1"])
    for d in range(1, num_factors - 1):
        args.append(factors[d])
        args.append(["b", f"r{d}", f"i{d}", f"r{d+1}"])

    # Add the last factor
    args.append(factors[-1])
    args.append(["b", f"r{num_factors - 1}", f"i{num_factors - 1}"])

    # Add the last factor
    args.append(["b", *(f"i{d}" for d in range(num_factors))])
    return oe.contract(*args)
