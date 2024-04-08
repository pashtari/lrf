import numpy as np

import cvxpy as cp
import pulp
from pulp import GLPK, PULP_CBC_CMD, COIN, HiGHS, COPT


def solve_least_absolute_errors(A, B, contruct_bound=(None, None), *args, **kwargs):
    M, N = B.shape
    _, P = A.shape

    lower_bound, upper_bound = contruct_bound

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


def solve_least_absolute_errors1(A, B, *args, **kwargs):
    M, N = B.shape
    _, P = A.shape

    # Create the problem
    prob = pulp.LpProblem("Least Absolute Errors", pulp.LpMinimize)

    # Create variables for X
    X = pulp.LpVariable.dicts("X", (range(P), range(N)), cat="Integer")

    # Create variables for C^+ and C^-
    C_plus = pulp.LpVariable.dicts(
        "C_plus", (range(M), range(N)), lowBound=0, cat="Continuous"
    )
    C_minus = pulp.LpVariable.dicts(
        "C_minus", (range(M), range(N)), lowBound=0, cat="Continuous"
    )

    # Objective function
    prob += pulp.lpSum([C_plus[m][n] + C_minus[m][n] for m in range(M) for n in range(N)])

    # Constraints
    for m in range(M):
        for n in range(N):
            prob += (
                B[m][n] - pulp.lpSum([A[m][p] * X[p][n] for p in range(P)])
                == C_plus[m][n] - C_minus[m][n],
                f"Constraint_pos_neg_{m}_{n}",
            )

    # Solve the problem
    prob.solve(*args, **kwargs)

    # Extract the solution for X
    X_opt = np.array([[X[p][n].varValue for n in range(N)] for p in range(P)])

    return X_opt


# def solve_least_absolute_errors2(A, B):
#     M, N = B.shape
#     _, P = A.shape

#     # Problem definition
#     prob = pulp.LpProblem("Least Absolute Errors", pulp.LpMinimize)

#     # Defining X variables
#     X = pulp.LpVariable.dicts("X", (range(P), range(N)), cat="Integer")

#     # Defining auxiliary variables for the absolute errors
#     E = pulp.LpVariable.dicts("E", (range(M), range(N)), lowBound=0)

#     # Objective function
#     prob += pulp.lpSum(E[m][n] for m in range(M) for n in range(N))

#     # Constraints for absolute values
#     for m in range(M):
#         for n in range(N):
#             prob += E[m][n] >= pulp.lpSum(A[m][p] * X[p][n] for p in range(P)) - B[m][n]
#             prob += E[m][n] >= B[m][n] - pulp.lpSum(A[m][p] * X[p][n] for p in range(P))

#     # Solve the problem
#     prob.solve(GLPK(msg=False))

#     # Extracting the solution
#     X_opt = np.zeros((P, N))
#     for p in range(P):
#         for n in range(N):
#             X_opt[p, n] = pulp.value(X[p][n])

#     return X_opt


# from scipy.optimize import linprog, milp, LinearConstraint


# def solve_least_absolute_errors3(A, B):
#     M, N = B.shape
#     _, P = A.shape

#     # Number of variables in the flattened solution X: P * N
#     # Total variables = X variables, C^+ and C^-
#     total_vars = P * N + 2 * M * N

#     # Objective function coefficients (only C^+ and C^- contribute to the cost)
#     c = np.concatenate((np.zeros(P * N), np.ones(2 * M * N)))

#     # Constraints
#     # Each difference |B_mn - (AX)_mn| is represented by C_mn^+ and C_mn^-
#     A_eq = np.zeros((M * N, total_vars))
#     b_eq = B.flatten()

#     # Constructing A_eq for enforcing B_mn - (AX)_mn = C_mn^+ - C_mn^-
#     for m in range(M):
#         for n in range(N):
#             # Index in the flattened B
#             idx = m * N + n
#             # Coefficients for AX part
#             A_eq[idx, n * P : (n + 1) * P] = A[m, :]
#             # Coefficient for C_mn^+
#             A_eq[idx, P * N + idx] = 1
#             # Coefficient for C_mn^-
#             A_eq[idx, P * N + M * N + idx] = -1

#     bounds = [(None, None)] * (P * N) + [(0, None)] * (2 * M * N)
#     integrality = np.concatenate((np.ones(P * N), np.zeros(2 * M * N)))

#     result = linprog(
#         c, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method="highs", integrality=integrality
#     )

#     if result.success:
#         # Extracting the X part of the solution
#         x_opt_flat = result.x[: P * N]
#         X_opt = x_opt_flat.reshape(N, P).T
#         return X_opt
#     else:
#         raise ValueError("MILP failed to find a solution: " + result.message)


import time
import matplotlib.pyplot as plt

# Example usage
A = np.random.randint(0, 10, size=(784, 10))
B = np.random.randint(0, 256, size=(784, 1))

X_star = solve_least_absolute_errors(
    A, B, contruct_bound=(0, 255), solver="GLPK_MI", warm_start=True
)
X_star1 = solve_least_absolute_errors1(A, B, GLPK(msg=False))

np.allclose(X_star, X_star1)


X = np.random.rand(5, 1)

mae = np.abs(B - A @ X).mean()
mae_star = np.abs(B - A @ X_star).mean()


# List of functions to profile
# solvers = [
#     GLPK(msg=False),
#     PULP_CBC_CMD(msg=False),
#     COIN(msg=False),
#     HiGHS(msg=False),
#     COPT(msg=False),
# ]

solvers = ["GLPK_MI", "SCIPY", "SCIP", "CBC"]

# Measure execution time of each function
execution_times = []
solver_names = []
for solver in solvers:
    start_time = time.time()
    solve_least_absolute_errors(A, B, solver=solver)  # Execute the function
    end_time = time.time()
    execution_times.append(end_time - start_time)
    solver_names.append(solver)

# Plotting the execution times
plt.figure(figsize=(10, 6))
plt.bar(solver_names, execution_times)
plt.xlabel("Functions")
plt.ylabel("Execution Time (seconds)")
plt.title("Solver Execution Time Comparison")
plt.show()


# import numpy as np


# def solve_modified_linear_program(c, A, b_prime, x_optimal):
#     """
#     Solves the modified linear program:
#     minimize c^T x
#     subject to Ax = b_prime
#                  x >= 0
#     by leveraging the optimal solution x_optimal from the initial problem.

#     Args:
#         c (np.ndarray): Coefficient vector for the objective function.
#         A (np.ndarray): Coefficient matrix for the constraints.
#         b_prime (np.ndarray): Modified right-hand side vector.
#         x_optimal (np.ndarray): Optimal solution from the initial problem.

#     Returns:
#         np.ndarray: Optimal solution for the modified problem.
#     """
#     M, _ = A.shape

#     # Compute the dual variables (reduced costs) for the initial problem
#     reduced_costs = c - A.T @ x_optimal

#     # Check if the modified right-hand side is feasible with the current solution
#     if np.all(A @ x_optimal == b_prime) and np.all(x_optimal >= 0):
#         return x_optimal

#     # Initialize the dual variables (initially set to zero)
#     dual_variables = np.zeros(M)

#     while True:
#         # Compute the dual objective function
#         dual_objective = b_prime @ dual_variables

#         # Compute the reduced costs for the modified problem
#         reduced_costs_prime = reduced_costs + A.T @ dual_variables

#         # Find the most negative reduced cost
#         entering_index = np.argmin(reduced_costs_prime)

#         # Check if the reduced cost is non-negative
#         if reduced_costs_prime[entering_index] >= 0:
#             return x_optimal  # The current solution is optimal for the modified problem

#         # Compute the direction vector for the dual simplex step
#         direction_vector = np.linalg.solve(A[:, entering_index], b_prime - A @ x_optimal)

#         # Compute the dual step length
#         dual_step_length = np.min(dual_variables / direction_vector)

#         # Update the dual variables
#         dual_variables -= dual_step_length * direction_vector

#         # Update the reduced costs for the modified problem
#         reduced_costs_prime = reduced_costs + A.T @ dual_variables

#         # Update the optimal solution for the modified problem
#         x_optimal += dual_step_length * direction_vector


# # Example usage:
# c = np.array([1, 2, 3])
# A = np.array([[1, 1, 1], [2, 1, 3]])
# b_prime = np.array([5, 8])
# x_optimal_initial = np.array([1, 2, 0])

# modified_solution = solve_modified_linear_program(c, A, b_prime, x_optimal_initial)
# print("Modified solution:", modified_solution)
