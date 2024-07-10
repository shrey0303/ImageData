import numpy as np
from helpers import P_matrix_in_blocks


def new_solution(
    X: np.array, y: list, L: int, u: int, sigma: int, h_u, eta
) -> tuple[list, np.array]:
    P = P_matrix_in_blocks(X, L, u, sigma)
    A = np.linalg.solve(np.eye(u) - (1 - eta) * P[3], np.eye(u))
    B = (1 - eta) * np.matmul(P[2], y[0:L]) + eta * h_u
    f_u = np.matmul(A, B)
    return (y[0:L], f_u)
