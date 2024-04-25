from typing import Optional, Callable

import numpy as np
import torch
from torch import Tensor
from torch import nn
from torch.nn.modules.utils import _pair
import cvxpy as cp
from joblib import Parallel, delayed

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
        project: Optional[Callable] = None,
        l2: float | tuple[float, float] = 0,
        l1_ratio: float = 0,
        eps: float = 1e-16,
    ):
        super().__init__()
        self.factor = _pair(factor)
        self.project = nn.Identity() if project is None else project
        self.l2 = _pair(l2)
        self.l1_ratio = l1_ratio
        self.eps = eps  # avoids division by zero

    def update_u(self, x: Tensor, u: Tensor, v: Tensor, l1: float, l2: float) -> Tensor:
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
                ur_new = (numerator + self.eps) / (denominator + self.eps)
                u_new[..., r : (r + 1)] = self.project(ur_new)
        else:
            numerator = soft_thresholding(a, l1)
            denominator = b + l2
            u_new = (numerator + self.eps) / (denominator + self.eps)
            u_new = self.project(u_new)

        return u_new

    def update_v(self, x: Tensor, u: Tensor, v: Tensor, l1: float, l2: float) -> Tensor:
        # x ≈ u @ t(v) --> v = ?
        return self.update_u(x.mT, v, u, l1, l2)

    def forward(self, x: Tensor, factors: tuple[Tensor, Tensor]) -> tuple[Tensor, Tensor]:
        u, v = factors
        *_, M, N = x.shape
        l1_u = self.l2[0] * self.l1_ratio * N
        l1_v = self.l2[1] * self.l1_ratio * M
        l2_u = self.l2[0] * (1 - self.l1_ratio) * N
        l2_v = self.l2[1] * (1 - self.l1_ratio) * M
        if 0 in self.factor:
            u = self.update_u(x, u, v, l1_u, l2_u)
        if 1 in self.factor:
            v = self.update_v(x, u, v, l1_v, l2_v)
        return u, v


def _least_absolute_errors(A, B, construct_bound=(None, None), *args, **kwargs):
    M, N = B.shape
    _, P = A.shape

    lower_bound, upper_bound = construct_bound

    # Create variable X
    X = cp.Variable((P, N), name="X", integer=True)

    # Create variables for C^+ and C^-
    C_plus = cp.Variable((M, N), name="C^+")
    C_minus = cp.Variable((M, N), name="C^-")

    # Objective function
    objective = cp.Minimize(cp.sum(C_plus + C_minus))

    # Constraints
    constraints = [A @ X - B == C_plus - C_minus, C_plus >= 0, C_minus >= 0]

    if lower_bound:
        constraints.append(A @ X >= lower_bound)

    if upper_bound:
        constraints.append(A @ X <= upper_bound)

    # Define the problem
    prob = cp.Problem(objective, constraints)

    # Solve
    prob.solve(*args, **kwargs)

    return X.value


def least_absolute_errors(A, B, *args, parallel=True, **kwargs):
    if parallel:

        def lae(b):
            return _least_absolute_errors(A, b.reshape(-1, 1), *args, **kwargs)

        results = Parallel(n_jobs=-1)(delayed(lae)(b) for b in B.T)
        return np.hstack(results).astype(A.dtype)
    else:
        return _least_absolute_errors(A, B, *args, **kwargs).astype(A.dtype)


class LeastAbsoluteErrors(nn.Module):
    "Mixed-integer linear programming solver for least absolute error."

    def __init__(self, *args, **kwargs):
        super().__init__()
        self.args = args
        self.kwargs = kwargs

    def update_u(self, x: Tensor, u: Tensor, v: Tensor) -> Tensor:
        # x ≈ u @ t(v) --> u = ?
        return self.update_v(x.mT, v, u)

    def update_v(self, x: Tensor, u: Tensor, v: Tensor) -> Tensor:
        # x ≈ u @ t(v) --> v = ?
        vt = least_absolute_errors(
            u.squeeze(0).numpy(), x.squeeze(0).numpy(), *self.args, **self.kwargs
        )
        vt = torch.from_numpy(vt).unsqueeze(0)
        return vt.mT

    def forward(self, x: Tensor, factors: tuple[Tensor, Tensor]) -> tuple[Tensor, Tensor]:
        u, v = factors
        u = self.update_u(x, u, v)
        v = self.update_v(x, u, v)
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
        self.init = SVDInit(rank=rank, project=self._project_to_int)
        self.solver = CoordinateDescent(project=self._project_to_int, **kwargs)
        # self.solver = LeastAbsoluteErrors(construct_bound=(0, 255), solver="GLPK_MI")
        self.verbose = verbose

    def _project_to_int(self, x: Tensor):
        x = torch.round(x)
        if self.bounds != (None, None):
            x = torch.clamp(x, self.bounds[0], self.bounds[1])
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
