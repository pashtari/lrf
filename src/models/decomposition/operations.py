import torch
import opt_einsum as oe


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

# def multi_mode_product1(tensor, matrices, modes=None, transpose=False):
#     modes = [*range(len(matrices))] if modes is None else modes
#     out = tensor
#     for mode, matrix in zip(modes, matrices):
#         out = mode_product(out, matrix, mode, transpose=transpose)

#     return out


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
