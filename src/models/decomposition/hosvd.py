import torch
from torch import nn
from torch.nn.modules.utils import _ntuple

from .operations import unfold, multi_mode_product


def hosvd(x, rank=None):
    ranks = _ntuple(x.ndim)(rank)
    factors = []
    for mode in range(x.ndim):
        unfold_x = unfold(x, mode)
        u, _, _ = torch.linalg.svd(unfold_x, full_matrices=False)
        if ranks[mode] is not None:
            u = u[:, : ranks[mode]]
        factors.append(u)

    core = multi_mode_product(x, factors, transpose=True)

    return core, factors


batch_hosvd = torch.vmap(hosvd, in_dims=(0, None))


# class HOSVD(nn.Module):
#     def __init__(self, rank=None):
#         super().__init__()
#         self.rank = rank

#     def fit(self, x):
#         return hosvd(x, rank=self.rank)

#     def reconstruct(self, core, factors):
#         return multi_mode_product(core, factors, transpose=False)

#     def forward(self, x):
#         core, factors = self.fit(x)
#         return self.reconstruct(core, factors)
