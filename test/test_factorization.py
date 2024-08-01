import torch

import lrf


def test_imf():
    x = torch.randint(0, 256, size=(1, 784, 192))
    imf = lrf.IMF(rank=5, num_iters=10, verbose=True)
    u, v = imf.decompose(x)
    return u, v


def test_hosvd_rank_upper_bounds():
    upper_bounds = lrf.hosvd_rank_upper_bounds([100, 5, 6])
    assert tuple(upper_bounds) == (30, 5, 6)


def test_hosvd():
    x = torch.rand(784, 8, 8, 3)
    core, factors = lrf.hosvd(x, rank=(49, 5, 5, 3))
    x_hat = lrf.multi_mode_product(core, factors)
    (x - x_hat).abs().mean()


def test_batched_hosvd():
    x = torch.rand(2, 784, 8, 8, 3)
    core, factors = lrf.batched_hosvd(x, rank=(49, 5, 5, 3))
    x_hat = lrf.batched_multi_mode_product(core, factors)
    (x - x_hat).abs().mean()


def test_tt_rank_upper_bounds():
    upper_bounds = lrf.tt_rank_upper_bounds([5, 20, 15, 10, 25])
    assert tuple(upper_bounds) == (5, 100, 250, 25)


def test_tt_rank_feasible_ranges():
    rank_ranges = lrf.tt_rank_feasible_ranges([5, 20, 15, 10, 25], 3)
    print(rank_ranges)


def test_ttd():
    x = torch.rand(784, 8, 8, 3)
    factors = lrf.ttd(x, rank=(49, 5, 3))
    x_hat = lrf.contract_tt(factors)
    (x - x_hat).abs().mean()


def test_batched_ttd():
    x = torch.rand(2, 784, 8, 8, 3)
    factors = lrf.batched_ttd(x, rank=(49, 5, 3))
    x_hat = lrf.batched_contract_tt(factors)
    (x - x_hat).abs().mean()


test_imf()
test_hosvd()
test_batched_hosvd()
test_hosvd_rank_upper_bounds()
test_ttd()
test_batched_ttd()
test_tt_rank_upper_bounds()
test_tt_rank_feasible_ranges()
