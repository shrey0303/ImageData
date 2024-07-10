import numpy as np
from helpers import (
    single_weight,
    unlabeled_part,
    labeled_part,
    smoothed_P_matrix_in_blocks,
    weight_matrix,
    P_matrix,
    harmonic_solution,
    label_entropy,
)


def partial_deriv_w(X: np.array, i: int, j: int, sigma: int) -> float:
    S = sum([(int(X[i, d]) - int(X[j, d])) ** 2 for d in range(X.shape[1])])
    temp = 2 * single_weight(X, i, j, sigma) * S
    power = sigma**3
    return temp / power


# quantity in expression (14)
def partial_deriv_p(
    X: np.array, i: int, j: int, sigma: int, W: np.array, P: np.array
) -> float:
    n = X.shape[0]
    S = 0
    sum = 0
    for index in range(n):
        S += partial_deriv_w(X, i, index, sigma)
        sum += W[i, index]
    result = partial_deriv_w(X, i, j, sigma) - P[i, j] * S
    return result / sum


def partial_deriv_P_tilde(
    X: np.array, sigma: int, eps: float, W: np.array, P: np.array
) -> np.array:
    n = X.shape[0]
    M = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            M[i, j] = partial_deriv_p(X, i, j, sigma, W, P)
    return (1 - eps) * M


def partial_deriv_P_tilde_in_blocks(
    X: np.array, sigma: int, eps: float, L: int, u: int, W: np.array, P: np.array
) -> tuple[np.array, np.array, np.array, np.array]:
    M = partial_deriv_P_tilde(X, sigma, eps, W, P)
    M_1 = M[0:L, 0:L]
    M_2 = M[0:L, L : L + u]
    M_3 = M[L : L + u, 0:L]
    M_4 = M[L : L + u, L : L + u]
    return (M_1, M_2, M_3, M_4)


# quantity in expression (13) df(u)/d(sigmad)
def derivative_vector(
    X: np.array,
    f: list,
    L: int,
    u: int,
    sigma: int,
    eps: float,
    temp,
    W: np.array,
    P: np.arry,
) -> np.array:
    a = np.matmul(
        partial_deriv_P_tilde_in_blocks(X, sigma, eps, L, u, W, P)[3],
        unlabeled_part(f, L),
    )
    b = np.matmul(
        partial_deriv_P_tilde_in_blocks(X, sigma, eps, L, u, W, P)[2],
        labeled_part(f, L),
    )
    v = np.matmul(temp, a + b)
    return v


# quantity in expression (12) dH/d(sigmad)
def compute_deriv(
    X: np.array,
    f: list,
    L: int,
    u: int,
    sigma: int,
    eps: float,
    temp,
    W: np.array,
    P: np.array,
) -> float:
    v = derivative_vector(X, f, L, u, sigma, eps, temp, W, P)
    return sum([np.log((1 - f[L + i]) / f[L + i]) * v[i] for i in range(u)]) / u


# quantities in expression (12) reunited in a vector
def compute_gradient(
    X: np.array, f: list, L: int, u: int, sigma: int, eps: float
) -> float:
    P_smoothed = smoothed_P_matrix_in_blocks(X, L, u, eps, sigma)
    temp = np.linalg.solve(np.eye(u) - P_smoothed[3], np.eye(u))
    W = weight_matrix(X, sigma)
    P = P_matrix(X, sigma)
    x = compute_deriv(X, f, L, u, sigma, eps, temp, W, P)
    return x


def gradient_descent(
    X: np.array,
    initial_sigma: int,
    max_iters: int,
    gamma,
    f: list,
    L: int,
    u: int,
    eps: float,
):
    sigma = initial_sigma
    for n_iter in range(max_iters):
        grad = compute_gradient(X, f, L, u, sigma, eps)
        sigma = sigma - gamma * grad
    return sigma


def grid_search(X: np.array, f: list, L: int, u: int, init: int, final: int) -> int:
    opt = init
    H = -np.infty
    for i in range(init, final - init):
        f = harmonic_solution(X, f, L, u, opt)
        temp = label_entropy(f, u, L)
        if temp < H:
            H = temp
            opt = i
    return opt
