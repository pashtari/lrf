import math
from typing import Optional, Callable

import torch
from torch import Tensor
from torch import nn
from torch.nn.modules.utils import _pair

from lrf.factorization.utils import relative_error, safe_divide, soft_thresholding


class RandInit(nn.Module):
    def __init__(
        self,
        rank: int,
        bounds: tuple[float, float],
    ) -> None:
        super().__init__()
        self.rank = rank
        self.bounds = bounds

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        (M, N), R = x.shape[-2:], self.rank
        u_shape = (*x.shape[:-2], M, R)
        v_shape = (*x.shape[:-2], N, R)
        alpha, beta = self.bounds
        u = torch.randint(alpha, beta + 1, (M, R)).expand(u_shape).float()
        v = torch.randint(alpha, beta + 1, (N, R)).expand(v_shape).float()
        return u, v


class SVDInit(nn.Module):
    def __init__(
        self,
        rank: int,
        num_levels: Optional[float] = None,
    ) -> None:
        super().__init__()
        self.rank = rank
        self.num_levels = num_levels

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        R = min(self.rank, *x.shape[-2:])
        u, s, v = torch.linalg.svd(x, full_matrices=False)
        u, s, v = u[..., :, :R], s[..., :R], v[..., :R, :]
        s = torch.sqrt(s)
        u = torch.einsum("...ir, ...r -> ...ir", u, s)
        v = torch.einsum("...rj, ...r -> ...jr", v, s)

        if self.rank > R:
            u = torch.nn.functional.pad(u, (0, self.rank - R))
            v = torch.nn.functional.pad(v, (0, self.rank - R))

        w0, w1 = torch.zeros_like(x[..., 0:1, 0:1]), torch.ones_like(x[..., 0:1, 0:1])

        if self.num_levels:
            scale_u = (
                u.amax(dim=(-2, -1), keepdim=True) - u.amin(dim=(-2, -1), keepdim=True)
            ) / self.num_levels
            scale_v = (
                v.amax(dim=(-2, -1), keepdim=True) - v.amin(dim=(-2, -1), keepdim=True)
            ) / self.num_levels
        else:
            scale_u = scale_v = 1

        u = u / scale_u
        v = v / scale_v
        w1 = (scale_u * scale_v) * w1

        w = torch.cat([w0, w1], dim=-2)
        return u, v, w


class CoordinateDescent(nn.Module):
    "Block coordinate descent update for linear least squares."

    def __init__(
        self,
        factor: int | tuple[int, int, int] = (0, 1, 2),
        project: Optional[Callable | tuple[Callable, Callable]] = None,
        l2: float | tuple[float, float] = 0,
        l1_ratio: float = 0,
        eps: float = 1e-16,
    ):
        super().__init__()
        self.factor = factor
        project = (nn.Identity(), nn.Identity()) if project is None else project
        self.project = _pair(project)
        self.l2 = _pair(l2)
        self.l1_ratio = l1_ratio
        self.eps = eps  # avoids division by zero

    def update_u(
        self,
        x: Tensor,
        u: Tensor,
        v: Tensor,
        w: Tensor,
        l1: float,
        l2: float,
        project: Callable,
    ) -> Tensor:
        # x ≈ w0 + w1 * u @ v.t --> u = ?
        w0, w1 = w.split(split_size=1, dim=-2)
        x = safe_divide(x - w0, w1, self.eps)
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
                ur_new = (numerator + self.eps) / (denominator + self.eps)
                u_new[..., r : (r + 1)] = project(ur_new)
        else:
            numerator = soft_thresholding(a, l1)
            denominator = b + l2
            u_new = (numerator + self.eps) / (denominator + self.eps)
            u_new = project(u_new)

        return u_new

    def update_v(
        self,
        x: Tensor,
        u: Tensor,
        v: Tensor,
        w: Tensor,
        l1: float,
        l2: float,
        project: Callable,
    ) -> Tensor:
        # x ≈ w0 + w1 * u @ v.t --> v = ?
        return self.update_u(x.mT, v, u, w, l1, l2, project)

    def update_w(self, x: Tensor, u: Tensor, v: Tensor, w: Tensor) -> Tensor:
        # x ≈ w0 + w1 * u @ v.t --> w = ?
        z = (u @ v.mT).flatten(-2, -1).unsqueeze(-1)
        a = torch.cat([torch.ones_like(z), z], dim=-1)
        b = x.flatten(-2, -1).unsqueeze(-1)
        w = torch.linalg.lstsq(a, b, rcond=1e-16).solution
        return w

    def forward(
        self, x: Tensor, factors: tuple[Tensor, Tensor, Tensor]
    ) -> tuple[Tensor, Tensor]:
        u, v, w = factors
        *_, M, N = x.shape
        l1_u = self.l2[0] * self.l1_ratio * N
        l1_v = self.l2[1] * self.l1_ratio * M
        l2_u = self.l2[0] * (1 - self.l1_ratio) * N
        l2_v = self.l2[1] * (1 - self.l1_ratio) * M
        if 0 in self.factor:
            u = self.update_u(x, u, v, w, l1_u, l2_u, self.project[0])
        if 1 in self.factor:
            v = self.update_v(x, u, v, w, l1_v, l2_v, self.project[1])
        if 2 in self.factor:
            w = self.update_w(x, u, v, w)
        return u, v, w


class IMF(nn.Module):
    """Implementation for integer matrix factorization (IMF).

    X ≈ w0 + w1 * (U @ V.T),
    U, V ∈ Z
    """

    def __init__(
        self,
        rank: Optional[int],
        num_iters: int = 10,
        bounds: tuple[None | float, None | float] = (None, None),
        num_levels: Optional[float] = None,
        verbose: bool = False,
        **kwargs,
    ) -> None:
        super().__init__()
        self.rank = rank
        self.num_iters = num_iters
        self.bounds = tuple(bounds)
        self.init = SVDInit(rank=rank, num_levels=num_levels)
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
        u, v, w = self.init(x)

        # iterate
        for it in range(1, self.num_iters + 1):
            if self.verbose:
                loss = self.loss(x, u, v, w)
                print(f"iter {it}: loss = {loss}")

            u, v, w = self.solver(x, [u, v, w], *args, **kwargs)

        return u, v, w

    @staticmethod
    def reconstruct(u: Tensor, v: Tensor, w: Optional[Tensor] = None) -> Tensor:
        out = u @ v.mT
        if w is None:
            return out
        else:
            w0, w1 = w.split(split_size=1, dim=-2)
            return w0 + w1 * out

    @staticmethod
    def loss(x: Tensor, u: Tensor, v: Tensor, w: Optional[Tensor] = None) -> Tensor:
        return relative_error(x, IMF.reconstruct(u, v, w))

    def forward(self, x: Tensor) -> Tensor:
        u, v, w = self.decompose(x)
        return self.reconstruct(u, v, w)
