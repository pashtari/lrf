import torch

from src.models import decomposition as decom


def test_hosvd():
    x = torch.rand(28, 28, 8, 8, 3)
    hosvd = decom.HOSVD(rank=(7, 7, 3, 3, 3))
    core, factors = hosvd.fit(x)
    x_hat = hosvd.reconstruct(core, factors)
    (x - x_hat).abs().mean()


test_hosvd()
