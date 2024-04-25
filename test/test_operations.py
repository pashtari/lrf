import torch

import lrf


def test_unfold():
    x = torch.rand(5, 6, 7, 8)
    y = lrf.unfold(x, 2)
    print("test_unfold: OK")


def test_mode_product():
    tensor = torch.rand(5, 6, 7, 8)
    matrix = torch.rand(3, 7)
    output1 = lrf.mode_product(tensor, matrix, 2, False)
    print("test_mode_product: OK")

    matrix = torch.rand(7, 3)
    output2 = lrf.mode_product(tensor, matrix, 2, True)
    print("test_mode_product: OK")


def test_multi_mode_product():
    tensor = torch.rand(5, 6, 7, 8)
    matrices = [torch.rand(3, 5), torch.rand(2, 6), torch.rand(4, 7), torch.rand(9, 8)]
    output1 = lrf.multi_mode_product(tensor, matrices, transpose=False)
    print("test_multi_mode_product: OK")


test_unfold()
test_mode_product()
test_multi_mode_product()
