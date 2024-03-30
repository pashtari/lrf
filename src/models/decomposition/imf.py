from typing import Any, Optional, Tuple, Union

import torch
from torch import Tensor
from torch import nn


def t(x: Tensor) -> Tensor:
    """Transpose a tensor, i.e. "b i j -> b j i"."""
    return x.transpose(-2, -1)


def dot(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
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

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        (M, N), R = x.shape[-2:], self.rank
        u_shape = (*x.shape[:-2], M, R)
        v_shape = (*x.shape[:-2], N, R)
        u = torch.randint(0, 256, (M, R)).expand(u_shape).to(torch.float32)
        v = torch.randint(0, 256, (N, R)).expand(v_shape).to(torch.float32)
        return u, v


class SVDInit(nn.Module):
    def __init__(self, rank: int, project: Any) -> None:
        super().__init__()
        self.rank = rank
        self.project = project

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor]:
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

    def update_u(self, x: torch.Tensor, u: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        # x ≈ u @ t(v) --> u = ?
        *_, M, N = x.shape
        if M >= N:
            u_new = x @ t(torch.linalg.pinv(v))
        else:
            a, b = x @ v, t(v) @ v
            u_new = torch.linalg.solve(b, t(a))
            u_new = t(u_new)

        return self.project(u_new)

    def update_v(self, x: Tensor, u: Tensor, v: Tensor) -> Tensor:
        # x ≈ u @ t(v) --> v = ?
        return self.update_u(t(x), v, u)

    def forward(self, x: Tensor, factors: Tuple[Tensor, Tensor]) -> Tuple[Tensor, Tensor]:
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

    def update_u(self, x: torch.Tensor, u: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        # x ≈ u @ t(v) --> u = ?
        a, b = x @ v, t(v) @ v
        g = a - u @ b
        η = (dot(g, g) + self.eps) / (dot(g, g @ b) + self.eps)
        η = η.unsqueeze(-1)
        u_new = self.project(u + η * g)
        return u_new

    def update_v(self, x: Tensor, u: Tensor, v: Tensor) -> Tensor:
        # x ≈ u @ t(v) --> v = ?
        return self.update_u(t(x), v, u)

    def forward(self, x: Tensor, factors: Tuple[Tensor, Tensor]) -> Tuple[Tensor, Tensor]:
        u, v = factors
        u = self.update_u(x, u, v)
        v = self.update_v(x, u, v)
        return u, v


class CoordinateDescent(nn.Module):
    "Block coordinate descent update for linear least squares."

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
        R = u.shape[-1]
        a, b = x @ v, t(v) @ v
        if R > 1:
            u_new = u.clone()
            for r in range(R):
                indices = [j for j in range(R) if j != r]
                term1 = a[..., r : (r + 1)].clone()
                term2 = u_new[..., indices].clone()
                term3 = b[..., indices, r : (r + 1)].clone()
                numerator = term1 - term2 @ term3 + self.eps
                denominator = b[..., r : (r + 1), r : (r + 1)].clone() + self.eps
                u_new[..., r : (r + 1)] = self.project(numerator / denominator).clone()
        else:
            numerator = a + self.eps
            denominator = b + self.eps
            u_new = self.project(numerator / denominator)

        return u_new

    def update_v(self, x: Tensor, u: Tensor, v: Tensor) -> Tensor:
        # x ≈ u @ t(v) --> v = ?
        return self.update_u(t(x), v, u)

    def forward(self, x: Tensor, factors: Tuple[Tensor, Tensor]) -> Tuple[Tensor, Tensor]:
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
        lower_bound: float = float("-inf"),
        upper_bound: float = float("inf"),
        verbose: bool = False,
        **kwargs,
    ) -> None:
        super().__init__()
        self.rank = rank
        self.num_iters = num_iters
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound
        self.init = SVDInit(rank=rank, project=self._project_to_int)
        # self.init = RandInit(rank=rank)
        self.solver = CoordinateDescent(project=self._project_to_int)
        # self.solver = LeastSquares(project=self._project_to_int)
        # self.solver = ProjectedGradient(project=self._project_to_int)
        self.verbose = verbose

    def _project_to_int(self, x: Tensor):
        x = torch.round(x)
        x = torch.clamp(x, self.lower_bound, self.upper_bound)
        return x

    def decompose(self, x: Tensor, *args, **kwargs) -> Tuple[Tensor, Tensor]:
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
        return u @ t(v)

    def loss(self, x: Tensor, u: Tensor, v: Tensor) -> Tensor:
        return relative_error(x, self.reconstruct(u, v))

    def forward(self, x: Tensor) -> Tensor:
        u, v = self.decompose(x)
        return self.reconstruct(u, v)
