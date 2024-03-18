import torch

from src.models import decomposition as decom


def test_hosvd_rank_upper_bounds():
    upper_bounds = decom.hosvd_rank_upper_bounds([100, 5, 6])
    assert tuple(upper_bounds) == (30, 5, 6)


test_hosvd_rank_upper_bounds()


def test_hosvd():
    x = torch.rand(784, 8, 8, 3)
    core, factors = decom.hosvd(x, rank=(49, 5, 5, 3))
    x_hat = decom.multi_mode_product(core, factors)
    (x - x_hat).abs().mean()


def test_batched_hosvd():
    x = torch.rand(2, 784, 8, 8, 3)
    core, factors = decom.batched_hosvd(x, rank=(49, 5, 5, 3))
    x_hat = decom.batched_multi_mode_product(core, factors)
    (x - x_hat).abs().mean()


def test_tt_rank_upper_bounds():
    upper_bounds = decom.tt_rank_upper_bounds([5, 20, 15, 10, 25])
    assert tuple(upper_bounds) == (5, 100, 250, 25)


def test_tt_rank_feasible_ranges():
    rank_ranges = decom.tt_rank_feasible_ranges([5, 20, 15, 10, 25], 3)
    print(rank_ranges)


test_tt_rank_feasible_ranges()


def test_ttd():
    x = torch.rand(784, 8, 8, 3)
    factors = decom.ttd(x, rank=(49, 5, 3))
    x_hat = decom.contract_tt(factors)
    (x - x_hat).abs().mean()


def test_batched_ttd():
    x = torch.rand(2, 784, 8, 8, 3)
    factors = decom.batched_ttd(x, rank=(49, 5, 3))
    x_hat = decom.batched_contract_tt(factors)
    (x - x_hat).abs().mean()


test_hosvd()
test_batched_hosvd()
test_tt_rank_upper_bounds()
test_ttd()
test_batched_ttd()
