import numpy as np

import os

os.environ["MKL_NUM_THREADS"] = "-1"
os.environ["OMP_NUM_THREADS"] = "-1"
os.environ["MKL_DYNAMIC"] = "FALSE"


from joblib import Parallel, delayed
import time


# Function for matrix multiplication
def matmul(u, v):
    return u @ v.T


def main():
    # Define the tensors (example dimensions)
    # List of tensor pairs
    u_list = [np.random.randn(1000, 100) for _ in range(100)]
    v_list = [np.random.randn(1000, 100) for _ in range(100)]
    tensor_pairs = list(zip(u_list, v_list))

    # Measure time for the for loop approach
    start_time = time.time()
    results_for_loop = []
    for u, v in tensor_pairs:
        result = matmul(u, v)
        results_for_loop.append(result)
    end_time = time.time()
    for_loop_time = end_time - start_time

    print(f"For loop approach time: {for_loop_time:.6f} seconds")

    # Measure time for the parallel approach using joblib
    start_time = time.time()
    results_parallel = Parallel(n_jobs=-1)(delayed(matmul)(u, v) for u, v in tensor_pairs)
    end_time = time.time()
    parallel_time = end_time - start_time

    print(f"Parallel approach time: {parallel_time:.6f} seconds")

    # Optional: Verify that the results are the same
    for i, (res_for, res_par) in enumerate(zip(results_for_loop, results_parallel)):
        if not np.allclose(res_for, res_par):
            print(f"Discrepancy found in result {i+1}")


if __name__ == "__main__":
    main()
