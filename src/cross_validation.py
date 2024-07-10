from helpers import classifier
from external_classifier import new_solution
from sklearn.metrics import accuracy_score
import numpy as np


def cross_valid_eta(
    X: np.array, y: list, L: int, u: int, sigma: int, h_u_continuous, increment: int
) -> float:
    init = 0
    (f_labeled, f_unlabeled) = new_solution(X, y, L, u, sigma, h_u_continuous, init)
    y_pred = classifier(f_unlabeled, 0.5)
    score = accuracy_score(y[L : L + u], y_pred)
    for i in np.arange(0.05, 1, increment):
        (f_labeled, f_unlabeled) = new_solution(X, y, L, u, sigma, h_u_continuous, i)
        y_pred = classifier(f_unlabeled, 0.5)
        temp = accuracy_score(y[L : L + u], y_pred)
        if score < temp:
            score = temp
            init = i
    return init
