from typing import Sequence, Optional

import torch
from torch import nn
from torch.nn.modules.utils import _ntuple
import opt_einsum as oe

from lrf.factorization.utils import prod


def hosvd_rank_upper_bounds(size: Sequence[int]):
    """
    Calculate the theoretical upper bounds of ranks for a tensor of arbitrary order
    in HOSVD.

    Args:
    - size (list of int): The size of the tensor.

    Returns:
    - tuple of int: The upper bounds of ranks for each mode of the tensor.
    """

    rank_upper_bounds = []
    for i, s in enumerate(size):
        # Calculate the product of dimensions excluding the i-th mode
        product_of_other_dimensions = prod(s for j, s in enumerate(size) if j != i)

        # The upper bound for the rank of the i-th mode is the minimum of
        # its size and the product of the other dimensions
        rank_upper_bound = min(s, product_of_other_dimensions)

        rank_upper_bounds.append(rank_upper_bound)

    return tuple(rank_upper_bounds)


def hosvd_rank_feasible_ranges(
    size: Sequence[int],
    com_ratio: float,
    rank: Optional[Sequence[Optional[int]]] = None,
):
    """
    Calculate feasible ranges of ranks for a tensor of arbitrary dimensions
    given a desired compression ratio.

    The function estimates lower and upper bounds for the ranks in a tensor
    train (TT) decomposition that would allow achieving a specified compression
    ratio.

    Args:
        size: Size of the tensor.
        com_ratio: Desired compression ratio.
        rank:

    Returns:
        list of tuples: Each tuple contains the lower and upper bounds
                        for the rank corresponding to each mode. The
                        bounds are estimated such that, if achievable, HOSVD
                        would not exceed the target storage size
                        implied by the specified compression ratio.
    """
    ranks = _ntuple(len(size))(rank)

    # Calculate the upper bounds for ranks
    initial_upper_bounds = hosvd_rank_upper_bounds(size)
    initial_upper_bounds = [r if r else u for r, u in zip(ranks, initial_upper_bounds)]

    # Calculate the lower bounds for ranks
    initial_lower_bounds = [r if r else 1 for r in ranks]

    # Calculate the total number of elements in the tensor
    total_elements = prod(size)

    # Target storage size based on the compression ratio
    target_storage = total_elements / com_ratio

    feasible_ranges = []
    for i, _ in enumerate(size):
        if ranks[i]:
            feasible_ranges.append((ranks[i], ranks[i]))
        else:
            # Lower bound estimation
            # Assume all other ranks at their maximum (upper bound) for the most conservative estimate
            storage_with_max_others = sum(
                initial_upper_bounds[j] * size[j] for j, _ in enumerate(size) if j != i
            )
            max_storage_current = target_storage - storage_with_max_others

            product_of_other_ranks = prod(
                initial_upper_bounds[j] for j, _ in enumerate(size) if j != i
            )
            lower_bound = max(
                1, int(max_storage_current / (size[i] + product_of_other_ranks))
            )

            # Upper bound estimation
            # Assume all other ranks at minimum (1) for the least conservative estimate
            storage_with_min_others = sum(
                initial_lower_bounds[j] * size[j] for j, _ in enumerate(size) if j != i
            )
            min_storage_current = target_storage - storage_with_min_others
            product_of_other_ranks = prod(
                initial_lower_bounds[j] for j, _ in enumerate(size) if j != i
            )
            upper_bound = min(
                initial_upper_bounds[i],
                int(min_storage_current / (size[i] + product_of_other_ranks)),
            )

            # Ensure logical ordering of bounds
            feasible_ranges.append((lower_bound, upper_bound))

    return feasible_ranges


def unfold(tensor, mode):
    shape = tensor.shape
    ndim = tensor.ndim
    tensor = tensor.permute(mode, *range(mode), *range(mode + 1, ndim))
    tensor = tensor.reshape(shape[mode], -1)
    return tensor


def mode_product(tensor, matrix, mode, transpose=False):
    tensor_inds = [*range(tensor.ndim)]
    matrix_inds = [mode, tensor.ndim] if transpose else [tensor.ndim, mode]
    output_inds = tensor_inds[:mode] + [tensor.ndim] + tensor_inds[mode + 1 :]
    out = oe.contract(tensor, tensor_inds, matrix, matrix_inds, output_inds)
    return out


batched_mode_product = torch.vmap(mode_product, in_dims=(0, 1, None, None))


def multi_mode_product(tensor, matrices, modes=None, transpose=False):
    modes = [*range(len(matrices))] if modes is None else modes
    args = []
    args.append(tensor)
    args.append([*range(tensor.ndim)])
    output_inds = []
    for mode in range(tensor.ndim):
        if mode in modes:
            matrix = matrices[modes.index(mode)]
            args.append(matrix)
            if transpose:
                args.append([mode, mode + tensor.ndim])
            else:
                args.append([mode + tensor.ndim, mode])
            output_inds.append(mode + tensor.ndim)
        else:
            output_inds.append(mode)

    args.append(output_inds)

    return oe.contract(*args)


def batched_multi_mode_product(tensor, matrices, modes=None, transpose=False):
    modes = [*range(len(matrices))] if modes is None else modes
    ndim = tensor.ndim - 1
    args = []
    args.append(tensor)
    args.append([-1, *range(ndim)])
    output_inds = [-1]
    for mode in range(ndim):
        if mode in modes:
            matrix = matrices[modes.index(mode)]
            args.append(matrix)
            if transpose:
                args.append([-1, mode, mode + ndim])
            else:
                args.append([-1, mode + ndim, mode])
            output_inds.append(mode + ndim)
        else:
            output_inds.append(mode)

    args.append(output_inds)

    return oe.contract(*args)


def hosvd(x, rank=None):
    ranks = _ntuple(x.ndim)(rank)
    assert len(ranks) == x.ndim
    factors = []
    for mode in range(x.ndim):
        unfold_x = unfold(x, mode)
        U, _, _ = torch.linalg.svd(unfold_x, full_matrices=False)
        if ranks[mode] is not None:
            U = U[:, : ranks[mode]]
        factors.append(U)

    core = multi_mode_product(x, factors, transpose=True)

    return core, factors


batched_hosvd = torch.vmap(hosvd, in_dims=(0,))


class HOSVD(nn.Module):
    def __init__(self, rank=None):
        super().__init__()
        self.rank = rank

    def decompose(self, x):
        return hosvd(x, rank=self.rank)

    def reconstruct(self, core, factors):
        return multi_mode_product(core, factors, transpose=False)

    def forward(self, x):
        core, factors = self.fit(x)
        return self.reconstruct(core, factors)
