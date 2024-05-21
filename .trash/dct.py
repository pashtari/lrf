import math
import torch


def create_dct_basis(N, **kwargs):
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


# Example usage
N = 4
dct_basis = create_dct_basis(N)
print(dct_basis)
print(dct_basis @ dct_basis.mT)

x = torch.rand(2, N, N)
y = torch.einsum("...mn, mk, nl -> ...kl", x, dct_basis, dct_basis)
xhat = torch.einsum("...mn, mk, nl -> ...kl", y, dct_basis.mT, dct_basis.mT)
print(f"{torch.allclose(x, xhat)}")
