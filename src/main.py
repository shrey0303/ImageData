import numpy as np
import plotly.graph_objects as go
from external_classifier import new_solution
from helpers import (
    laplace_smoothing,
    harmonic_solution,
    create_sample,
    classifier,
    classifier_thresold,
)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from tqdm import tqdm


def simulation(parameters: dict, data: list, sizes: list[int]) -> go.Figure:
    S = parameters["S"]
    N = parameters["nb_points"]
    p = parameters["nb_pixels"]
    q = parameters["cmn_q"]
    sigma = parameters["sigma"]
    (X, f) = data
    for i in range(len(f)):
        f[i] -= 1

    X = np.array(X)
    accuracy_thr = []
    accuracy_cmn = []
    accuracy_lr = []
    accuracy_ext = []

    for L in tqdm(sizes):
        u = N - L
        X_spl = np.zeros((S, N, p))
        f_spl = np.zeros((S, N))
        f_spl_labeled = np.zeros((S, L))
        f_spl_unlabeled = np.zeros((S, u))
        f_u_classified = np.zeros((S, u))
        f_u_thr = np.zeros((S, u))
        f_spl_labeled_ext = np.zeros((S, L))
        f_spl_unlabeled_ext = np.zeros((S, u))
        f_u_classified_ext = np.zeros((S, u))
        for i in range(S):
            (X_spl[i], f_spl[i]) = create_sample(X, f, N, L, u, p)
            q = laplace_smoothing(f_spl[i][0:L])

            # harmonic solution
            (f_spl_labeled[i], f_spl_unlabeled[i]) = harmonic_solution(
                X_spl[i], f_spl[i], L, u, sigma
            )
            f_u_classified[i] = classifier(f_spl_unlabeled[i], q)
            accuracy_cmn.append(
                (accuracy_score(f_spl[i][L : L + u], f_u_classified[i]))
            )

            # thresold method
            f_u_thr[i] = classifier_thresold(f_spl_unlabeled[i])
            accuracy_thr.append((accuracy_score(f_spl[i][L : L + u], f_u_thr[i])))

            # logistic regression
            logreg = LogisticRegression()
            logreg.fit(X_spl[i][0:L], f_spl[i][0:L])
            y_pred = logreg.predict(X_spl[i][L : L + u])
            accuracy_lr.append(accuracy_score(f_spl[i][L : L + u], y_pred))

            # external classifier (label propagation + logistic regression)
            y_pred_continuous = logreg.predict_proba(X_spl[i][L : L + u])[:, 1]
            (f_spl_labeled_ext[i], f_spl_unlabeled_ext[i]) = new_solution(
                X_spl[i], f_spl[i], L, u, sigma, y_pred_continuous, eta=0.1
            )
            f_u_classified_ext[i] = classifier(f_spl_unlabeled_ext[i], q)
            accuracy_ext.append(
                accuracy_score(f_spl[i][L : L + u], f_u_classified_ext[i])
            )
    fig = go.Figure()
    methods = [
        "label propagation",
        "logistic regression",
        "logistic regression",
        "thresold method",
    ]
    for acc, method in zip(
        [accuracy_cmn, accuracy_lr, accuracy_ext, accuracy_thr], methods
    ):
        fig.add_trace(go.Scatter(x=sizes, y=acc, name=method))
    fig.update_layout(
        legend_title_text="Method",
        title="Compared method accuracies",
        xaxis_title="number of labeled images",
        yaxis_title="method accuracy",
    )
    return fig
