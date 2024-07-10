import numpy as np
import random as rd


def single_weight(X: np.array, i: int, j: int, sigma: int) -> float:
    # if X has shape (n, m)
    m = X.shape[1]
    s = sigma**2
    S = sum(list(map(lambda d: (int(X[i, d]) - int(X[j, d])) ** 2 / s, range(m))))
    E = np.exp(-S)
    return E


def weight_matrix(X: np.array, sigma: int) -> np.array:
    n = X.shape[0]
    W = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            W[i, j] = single_weight(X, i, j, sigma)
    return W


def quadratic_energy(X: np.array, f: list, sigma: int) -> float:
    L = []
    n = X.shape[0]
    m = X.shape[1]
    W = weight_matrix(X, sigma)
    L = [W[i, j] * (f[i] - f[j]) ** 2 for i in range(n) for j in range(m)]
    S = L.sum()
    E = 0.5 * S
    return E


def magic_entropy_coeff(z: float) -> float:
    return -z * np.log(z) - ((1 - z) * np.log(1 - z))


def label_entropy(f: list, u: int, L: int):
    return list(map(lambda i: magic_entropy_coeff(f[L + i]), range(u))).sum() / u


def labeled_part(f: list, L: int) -> list:
    return f[:L]


def unlabeled_part(f: list, L: int) -> list:
    return f[L:]


def weight_in_blocks(
    X: np.array, L: int, u: int, sigma: int
) -> tuple[np.array, np.array, np.array, np.array]:
    W = weight_matrix(X, sigma)
    W_1 = W[0:L, 0:L]
    W_2 = W[0:L, L : L + u]
    W_3 = W[L : L + u, 0:L]
    W_4 = W[L : L + u, L : L + u]
    return (W_1, W_2, W_3, W_4)


def diagonal_matrix(X: np.array, sigma: int) -> np.array:
    n = X.shape[0]
    d = []
    W = weight_matrix(X, sigma)
    for i in range(n):
        s = W[i, :].sum()
        d.append(s)
    D = np.diag(d)
    return D


def diagonal_in_blocks(
    X: np.array, L: int, u: int, sigma: int
) -> tuple[np.array, np.array, np.array, np.array]:
    D = diagonal_matrix(X, sigma)
    D_1 = D[0:L, 0:L]
    D_2 = D[0:L, L : L + u]
    D_3 = D[L : L + u, 0:L]
    D_4 = D[L : L + u, L : L + u]
    return (D_1, D_2, D_3, D_4)


def laplacian(X: np.array, sigma: int) -> np.array:
    return diagonal_matrix(X, sigma) - weight_matrix(X, sigma)


def P_matrix(X, sigma: int) -> np.array:
    W = weight_matrix(X, sigma)
    D = diagonal_matrix(X, sigma)
    return np.linalg.solve(D, W)


def P_matrix_in_blocks(
    X: np.array, L: int, u: int, sigma: int
) -> tuple[np.array, np.array, np.array, np.array]:
    D = P_matrix(X, sigma)
    D_1 = D[0:L, 0:L]
    D_2 = D[0:L, L : L + u]
    D_3 = D[L : L + u, 0:L]
    D_4 = D[L : L + u, L : L + u]
    return (D_1, D_2, D_3, D_4)


def smoothed_P_matrix(X: np.array, eps: float, sigma: int) -> np.array:
    n = X.shape[0]
    P = P_matrix(X, sigma)
    U = np.zeros((n, n))
    return eps * U + (1 - eps) * P


def smoothed_P_matrix_in_blocks(
    X: np.array, L: int, u: int, eps: float, sigma: int
) -> tuple[np.array, np.array, np.array, np.array]:
    D = smoothed_P_matrix(X, eps, sigma)
    D_1 = D[0:L, 0:L]
    D_2 = D[0:L, L : L + u]
    D_3 = D[L : L + u, 0:L]
    D_4 = D[L : L + u, L : L + u]
    return (D_1, D_2, D_3, D_4)


def harmonic_solution_smoothed(
    X: np.array, y: list, L: int, u: int, sigma: int, eps: float
) -> tuple[list, list]:
    f_labeled = y[:L]
    temp = np.linalg.solve(
        np.eye(u) - smoothed_P_matrix_in_blocks(X, L, u, eps, sigma)[3], np.eye(u)
    )
    f_unlabeled = np.matmul(
        np.matmul(temp, smoothed_P_matrix_in_blocks(X, L, u, eps, sigma)[2]), f_labeled
    )
    return (f_labeled, f_unlabeled)


def harmonic_solution(
    X: np.array, y: list, L: int, u: int, sigma: int
) -> tuple[list, list]:
    W_b = weight_in_blocks(X, L, u, sigma)
    f_labeled = y[:L]
    temp = np.linalg.solve(diagonal_in_blocks(X, L, u, sigma)[3] - W_b[3], np.eye(u))
    f_unlabeled = np.matmul(np.matmul(temp, W_b[2]), f_labeled)
    return (f_labeled, f_unlabeled)


def create_sample(
    X: np.array, f: list, N: int, L: int, u: int, p: int
) -> tuple[np.array, list]:
    X_spl = np.zeros((N, p))
    f_spl = []
    unlabeled = 1
    increment = 0
    spl = rd.sample(range(N), L)

    # fill labeled part of X_spl
    for j in range(L):
        X_spl[j] = X[spl[j]]
        f_spl.append(f[spl[j]])

    # fill unlabeled part of X_spl
    for j in range(N):
        for k in range(L):
            if (j) == spl[k]:
                unlabeled = 0
                increment += 1
        if unlabeled == 1:
            X_spl[L + j - increment] = X[j]
            f_spl.append(f[j])
        unlabeled = 1
    return (X_spl, f_spl)


def classifier(f_unlabeled: list, q: float) -> np.array:
    S = sum(f_unlabeled)
    u = len(f_unlabeled)
    f_u_classified = np.zeros(u)
    for i in range(u):
        if q * f_unlabeled[i] / S > (1 - q) * (1 - f_unlabeled[i]) / (u - S):
            f_u_classified[i] = 1
    return f_u_classified


def classifier_thresold(f_unlabeled: list) -> np.array:
    u = len(f_unlabeled)
    f_u_classified = np.zeros(u)
    for i in range(u):
        if f_unlabeled[i] > 1 / 2:
            f_u_classified[i] = 1
    return f_u_classified


def laplace_smoothing(f_labeled) -> float:
    n_1 = len([x for x in f_labeled if x == 1])
    q = (n_1 + 1) / (len(f_labeled) + 2)
    return q
