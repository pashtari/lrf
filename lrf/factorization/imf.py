import math
from typing import Optional, Callable

import torch
from torch import Tensor
from torch import nn
from torch.nn.modules.utils import _pair
from einops import rearrange

from lrf.factorization.utils import relative_error, soft_thresholding


class RandInit(nn.Module):
    def __init__(
        self,
        rank: int,
    ) -> None:
        super().__init__()
        self.rank = rank

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        (M, N), R = x.shape[-2:], self.rank
        u_shape = (*x.shape[:-2], M, R)
        v_shape = (*x.shape[:-2], N, R)
        u = torch.randint(-128, 128, (M, R)).expand(u_shape).float()
        v = torch.randint(-128, 128, (N, R)).expand(v_shape).float()
        return u, v


class DCTInit(nn.Module):
    def __init__(
        self,
        rank: int,
        project: Optional[tuple[Callable, Callable]] = None,
    ) -> None:
        super().__init__()
        self.rank = rank
        self.project = nn.Identity() if project is None else project
        self.patch_zie = (8, 8)

    def create_dct_basis(self, N, **kwargs):
        """Create the NxN DCT basis matrix."""

        # Create a tensor for the indices
        n = torch.arange(N, **kwargs).view(1, N)

        # Compute the basis matrix
        theta = torch.pi * (2 * n + 1) * n.mT / (2 * N)
        basis = torch.cos(theta)

        # Apply the normalization factors
        basis[0, :] *= 1 / math.sqrt(N)
        basis[1:, :] *= math.sqrt(2 / N)

        return basis

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        R = self.rank
        p, q = self.patch_zie
        y = rearrange(x, "... n (c p q) -> ... n c p q", p=p, q=q)
        ch, cw = self.create_dct_basis(p), self.create_dct_basis(q)
        u = torch.einsum("... pq, pk, ql -> ... kl", y, ch, cw)
        # (x.mT @ ch).mT @ cw
        K, L = max(1, int(math.sqrt(R))), max(1, int(math.sqrt(R)))
        u = u[..., :K, :L]
        u = rearrange(u, "... c p q -> ...(c p q)")

        ch, cw = ch[:, :K], cw[:, :L]
        v = torch.einsum("pk, ql -> pqkl", ch, cw)
        # torch.einsum("...pq, pk, ql -> ... kl", x, ch, cw)
        v = rearrange(v, "p q k l -> (p q) (k l)")

        s = u.norm(dim=-2, keepdim=True)
        u = u / (s + 1e-8)
        s = torch.sqrt(s)
        u = torch.einsum("...ir, ...r -> ...ir", u, s)
        v = torch.einsum("...jr, ...r -> ...jr", v, s)
        u, v = u.squeeze(1), v.squeeze(1)
        return self.project(u), self.project(v)


class SVDInit(nn.Module):
    def __init__(
        self, rank: int, project: Optional[tuple[Callable, Callable]] = None
    ) -> None:
        super().__init__()
        self.rank = rank
        self.project = nn.Identity() if project is None else project

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        R = self.rank
        u, s, v = torch.linalg.svd(x, full_matrices=False)
        u, s, v = u[..., :, :R], s[..., :R], v[..., :R, :]
        s = torch.sqrt(s)
        u = torch.einsum("...ir, ...r -> ...ir", u, s)
        v = torch.einsum("...rj, ...r -> ...jr", v, s)
        return self.project(u), self.project(v)


class CoordinateDescent(nn.Module):
    "Block coordinate descent update for linear least squares."

    def __init__(
        self,
        factor: int | tuple[int, int] = (0, 1),
        project: Optional[Callable | tuple[Callable, Callable]] = None,
        l2: float | tuple[float, float] = 0,
        l1_ratio: float = 0,
        eps: float = 1e-16,
    ):
        super().__init__()
        self.factor = _pair(factor)
        project = (nn.Identity(), nn.Identity()) if project is None else project
        self.project = _pair(project)
        self.l2 = _pair(l2)
        self.l1_ratio = l1_ratio
        self.eps = eps  # avoids division by zero

    def update_u(
        self, x: Tensor, u: Tensor, v: Tensor, l1: float, l2: float, project: Callable
    ) -> Tensor:
        # x ≈ u @ t(v) --> u = ?
        R = u.shape[-1]
        a, b = x @ v, v.mT @ v
        if R > 1:
            u_new = u.clone()
            for r in range(R):
                indices = [j for j in range(R) if j != r]
                term1 = a[..., r : (r + 1)]
                uu = u_new[..., indices]
                bb = b[..., indices, r : (r + 1)]
                term2 = uu @ bb
                numerator = soft_thresholding(term1 - term2, l1)
                denominator = b[..., r : (r + 1), r : (r + 1)] + l2
                ur_new = numerator / (denominator + self.eps)
                u_new[..., r : (r + 1)] = project(ur_new)
        else:
            numerator = soft_thresholding(a, l1)
            denominator = b + l2
            u_new = numerator / (denominator + self.eps)
            u_new = project(u_new)

        return u_new

    def update_v(
        self, x: Tensor, u: Tensor, v: Tensor, l1: float, l2: float, project: Callable
    ) -> Tensor:
        # x ≈ u @ t(v) --> v = ?
        return self.update_u(x.mT, v, u, l1, l2, project)

    def forward(self, x: Tensor, factors: tuple[Tensor, Tensor]) -> tuple[Tensor, Tensor]:
        u, v = factors
        *_, M, N = x.shape
        l1_u = self.l2[0] * self.l1_ratio * N
        l1_v = self.l2[1] * self.l1_ratio * M
        l2_u = self.l2[0] * (1 - self.l1_ratio) * N
        l2_v = self.l2[1] * (1 - self.l1_ratio) * M
        if 0 in self.factor:
            u = self.update_u(x, u, v, l1_u, l2_u, self.project[0])
        if 1 in self.factor:
            v = self.update_v(x, u, v, l1_v, l2_v, self.project[1])
        return u, v


class IMF(nn.Module):
    """Implementation for integer matrix factorization (IMF).

    X ≈ U t(V),
    U, V ∈ Z
    """

    def __init__(
        self,
        rank: Optional[int],
        num_iters: int = 10,
        bounds: tuple[None | float, None | float] = (None, None),
        verbose: bool = False,
        **kwargs,
    ) -> None:
        super().__init__()
        self.rank = rank
        self.num_iters = num_iters
        self.bounds = tuple(bounds)
        self.init = SVDInit(rank=rank, project=self._project)
        self.solver = CoordinateDescent(project=self._project, **kwargs)
        self.verbose = verbose

    def _project(self, x: Tensor):
        x = torch.round(x)
        if self.bounds != (None, None):
            x = torch.clamp(x, math.ceil(self.bounds[0]), math.floor(self.bounds[1]))
        return x

    def decompose(self, x: Tensor, *args, **kwargs) -> tuple[Tensor, Tensor]:
        # x: B × M × N

        # convert x to float
        x = x.float()

        # initialize
        u, v = self.init(x)

        # iterate
        for it in range(1, self.num_iters + 1):
            if self.verbose:
                loss = self.loss(x, u, v)
                print(f"iter {it}: loss = {loss}")

            u, v = self.solver(x, [u, v], *args, **kwargs)

        return u, v

    def reconstruct(self, u: Tensor, v: Tensor) -> Tensor:
        return u @ v.mT

    def loss(self, x: Tensor, u: Tensor, v: Tensor) -> Tensor:
        return relative_error(x, self.reconstruct(u, v))

    def forward(self, x: Tensor) -> Tensor:
        u, v = self.decompose(x)
        return self.reconstruct(u, v)
