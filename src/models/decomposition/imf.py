from typing import Any, Optional, Sequence, Callable

import numpy as np
from joblib import Parallel, delayed
import cvxpy as cp
import torch
from torch import Tensor
from torch import nn


def dot(x: Tensor, y: Tensor) -> Tensor:
    out = (x * y).flatten(-2).sum(-1, keepdim=True)
    return out


def relative_error(tensor1, tensor2, epsilon: float = 1e-16):
    numerator = torch.norm(tensor1 - tensor2, p=2, dim=(-2, -1))
    denominator = torch.norm(tensor1, p=2, dim=(-2, -1))
    return numerator / (denominator + epsilon)


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
        u = torch.randint(-128, 128, (M, R)).expand(u_shape).to(torch.float32)
        v = torch.randint(-128, 128, (N, R)).expand(v_shape).to(torch.float32)
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


class LeastSquares(nn.Module):
    """Least Squares."""

    def __init__(
        self,
        project: Optional[Any] = None,
        eps: float = 1e-8,
    ):
        super().__init__()
        self.project = nn.Identity() if project is None else project
        self.eps = eps

    def update_u(self, x: Tensor, u: Tensor, v: Tensor) -> Tensor:
        # x ≈ u @ t(v) --> u = ?
        *_, M, N = x.shape
        if M >= N:
            u_new = x @ t(torch.linalg.pinv(v))
        else:
            a, b = x @ v, v.mT @ v
            u_new = torch.linalg.solve(b, a.mT)
            u_new = u_new.mT

        return self.project(u_new)

    def update_v(self, x: Tensor, u: Tensor, v: Tensor) -> Tensor:
        # x ≈ u @ t(v) --> v = ?
        return self.update_u(x.mT, v, u)

    def forward(self, x: Tensor, factors: tuple[Tensor, Tensor]) -> tuple[Tensor, Tensor]:
        u, v = factors
        u = self.update_u(x, u, v)
        v = self.update_v(x, u, v)
        return u, v


class ProjectedGradient(nn.Module):
    "Projected gradient descent with line search for linear least squares."

    def __init__(
        self,
        project: Optional[Any] = None,
        eps: float = 1e-8,
    ):
        super().__init__()
        self.project = nn.Identity() if project is None else project
        self.eps = eps

    def update_u(self, x: Tensor, u: Tensor, v: Tensor) -> Tensor:
        # x ≈ u @ t(v) --> u = ?
        a, b = x @ v, v.mT @ v
        g = a - u @ b
        η = (dot(g, g) + self.eps) / (dot(g, g @ b) + self.eps)
        η = η.unsqueeze(-1)
        u_new = self.project(u + η * g)
        return u_new

    def update_v(self, x: Tensor, u: Tensor, v: Tensor) -> Tensor:
        # x ≈ u @ t(v) --> v = ?
        return self.update_u(x.mT, v, u)

    def forward(self, x: Tensor, factors: tuple[Tensor, Tensor]) -> tuple[Tensor, Tensor]:
        u, v = factors
        u = self.update_u(x, u, v)
        v = self.update_v(x, u, v)
        return u, v


class CoordinateDescent(nn.Module):
    "Block coordinate descent update for linear least squares."

    def __init__(
        self,
        project: Optional[tuple[Callable, Callable]] = None,
        eps: float = 1e-8,
    ):
        super().__init__()
        self.project = nn.Identity() if project is None else project
        self.eps = eps

    def update_u(self, x: Tensor, u: Tensor, v: Tensor) -> Tensor:
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
                numerator = term1 - term2
                denominator = b[..., r : (r + 1), r : (r + 1)]
                ur_new = (numerator + self.eps) / (denominator + self.eps)
                u_new[..., r : (r + 1)] = self.project(ur_new)
        else:
            numerator = a + self.eps
            denominator = b + self.eps
            u_new = self.project(numerator / denominator)

        return u_new

    def update_v(self, x: Tensor, u: Tensor, v: Tensor) -> Tensor:
        # x ≈ u @ t(v) --> v = ?
        return self.update_u(x.mT, v, u)

    def forward(self, x: Tensor, factors: tuple[Tensor, Tensor]) -> tuple[Tensor, Tensor]:
        u, v = factors
        u = self.update_u(x, u, v)
        v = self.update_v(x, u, v)
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
        # self.init = RandInit(rank=rank)
        self.solver = CoordinateDescent(project=self._project_to_int)
        # self.solver = LeastSquares(project=self._project_to_int)
        # self.solver = ProjectedGradient(project=self._project_to_int)
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
        x = x.to(torch.float32)

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
