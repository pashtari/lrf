import torch
import tensorly as tl

tl.set_backend("pytorch")

import factorization as fact


def test_unfold():
    x = torch.rand(5, 6, 7, 8)
    y1 = fact.unfold(x, 2)
    y2 = tl.unfold(x, 2)
    print(f"test_unfold: {torch.all(y1 == y2)}")


def test_mode_product():
    tensor = torch.rand(5, 6, 7, 8)
    matrix = torch.rand(3, 7)
    output1 = fact.mode_product(tensor, matrix, 2, False)
    output2 = tl.tenalg.mode_dot(tensor, matrix, 2, False)
    print(f"test_mode_product: {torch.all(output1 == output2)}")

    matrix = torch.rand(7, 3)
    output1 = fact.mode_product(tensor, matrix, 2, True)
    output2 = tl.tenalg.mode_dot(tensor, matrix, 2, True)
    print(f"test_mode_product: {torch.all(output1 == output2)}")


def test_multi_mode_product():
    tensor = torch.rand(5, 6, 7, 8)
    matrices = [torch.rand(3, 5), torch.rand(2, 6), torch.rand(4, 7), torch.rand(9, 8)]
    output1 = fact.multi_mode_product(tensor, matrices, transpose=False)
    output2 = tl.tenalg.multi_mode_dot(tensor, matrices, transpose=False)
    print(f"test_multi_mode_product: {torch.allclose(output1, output2)}")


test_unfold()
test_mode_product()
test_multi_mode_product()


# from timeit import timeit

# tensor = torch.rand(50, 60, 70, 90)
# matrices = [torch.rand(2, 50), torch.rand(10, 60), torch.rand(3, 70), torch.rand(2, 90)]


# total_time = timeit(
#     "fact.multi_mode_product(tensor, matrices, modes=None, transpose=False)",
#     number=100,
#     globals=globals(),
# )
# print(f"Average time is {total_time*1000 / 100:.2f} ms")


# total_time = timeit(
#     "tl.tenalg.multi_mode_dot(tensor, matrices, modes=None, transpose=False)",
#     number=100,
#     globals=globals(),
# )
# print(f"Average time is {total_time*1000 / 100:.2f} ms")
