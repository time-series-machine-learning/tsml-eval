"""Code for tuning clusterer parameters used in the publication."""

import sys

import numpy as np
from aeon.clustering import TimeSeriesKMeans
from sklearn.metrics import davies_bouldin_score


# used for dtw and wdtw primarily
def _tune_window(method, train_X, n_clusters):  # pragma: no cover
    best_w = 0
    best_score = sys.float_info.max
    for w in np.arange(0.0, 0.2, 0.01):
        cls = TimeSeriesKMeans(
            distance=method, distance_params={"window": w}, n_clusters=n_clusters
        )
        cls.fit(train_X)
        preds = cls.predict(train_X)
        clusters = len(np.unique(preds))
        if clusters <= 1:
            score = sys.float_info.max
        else:
            score = davies_bouldin_score(train_X, preds)
        print(f" Number of clusters = {clusters} window = {w} score = {score}")
        if score < best_score:
            best_score = score
            best_w = w
    print("best window =", best_w, " with score ", best_score)
    return best_w


def _tune_msm(train_X, n_clusters):  # pragma: no cover
    best_c = 0
    best_score = sys.float_info.max
    for c in np.arange(0.0, 5.0, 0.25):
        cls = TimeSeriesKMeans(
            distance="msm", distance_params={"c": c}, n_clusters=n_clusters
        )
        cls.fit(train_X)
        preds = cls.predict(train_X)
        clusters = len(np.unique(preds))
        if clusters <= 1:
            score = sys.float_info.max
        else:
            score = davies_bouldin_score(train_X, preds)
        print(f" Number of clusters = {clusters} c parameter = {c} score = {score}")
        if score < best_score:
            best_score = score
            best_c = c
    print("best c =", best_c, " with score ", best_score)
    return best_c


def _tune_wdtw(train_X, n_clusters):  # pragma: no cover
    best_g = 0
    best_score = sys.float_info.max
    for g in np.arange(0.0, 1.0, 0.05):
        cls = TimeSeriesKMeans(
            distance="wdtw", distance_params={"g": g}, n_clusters=n_clusters
        )
        cls.fit(train_X)
        preds = cls.predict(train_X)
        clusters = len(np.unique(preds))
        if clusters <= 1:
            score = sys.float_info.max
        else:
            score = davies_bouldin_score(train_X, preds)
        print(f" Number of clusters = {clusters} g parameter = {g} score = {score}")
        if score < best_score:
            best_score = score
            best_g = g
    print("best g =", best_g, " with score ", best_score)
    return best_g


def _tune_twe(train_X, n_clusters):  # pragma: no cover
    best_nu = 0
    best_lambda = 0
    best_score = sys.float_info.max
    for nu in np.arange(0.0, 1.0, 0.25):
        for lam in np.arange(0.0, 1.0, 0.2):
            cls = TimeSeriesKMeans(
                distance="twe",
                distance_params={"nu": nu, "lmbda": lam},
                n_clusters=n_clusters,
            )
            cls.fit(train_X)
            preds = cls.predict(train_X)
            clusters = len(np.unique(preds))
            if clusters <= 1:
                score = sys.float_info.max
            else:
                score = davies_bouldin_score(train_X, preds)
            print(
                f" Number of clusters = {clusters} nu param = {nu} lambda para "
                f"= {lam} score = {score}"
            )  #
            # noqa
            if score < best_score:
                best_score = score
                best_nu = nu
                best_lambda = lam
    print("best nu =", best_nu, f" lambda = {best_lambda} score ", best_score)  # noqa
    return best_nu, best_lambda


def _tune_erp(train_X, n_clusters):  # pragma: no cover
    best_g = 0
    best_score = sys.float_info.max
    for g in np.arange(0.0, 2.0, 0.2):
        cls = TimeSeriesKMeans(
            distance="erp", distance_params={"g": g}, n_clusters=n_clusters
        )
        cls.fit(train_X)
        preds = cls.predict(train_X)
        clusters = len(np.unique(preds))
        if clusters <= 1:
            score = sys.float_info.max
        else:
            score = davies_bouldin_score(train_X, preds)
        print(f" Number of clusters ={clusters} g parameter = {g} score  = {score}")
        if score < best_score:
            best_score = score
            best_g = g
    print("best g =", best_g, " with score ", best_score)
    return best_g


def _tune_edr(train_X, n_clusters):  # pragma: no cover
    best_e = 0
    best_score = sys.float_info.max
    for e in np.arange(0.0, 0.2, 0.01):
        cls = TimeSeriesKMeans(
            distance="edr", distance_params={"epsilon": e}, n_clusters=n_clusters
        )
        cls.fit(train_X)
        preds = cls.predict(train_X)
        clusters = len(np.unique(preds))
        if clusters <= 1:
            score = sys.float_info.max
        else:
            score = davies_bouldin_score(train_X, preds)
        print(
            f" Number of clusters = {clusters} epsilon parameter = {e} score = {score}"
        )
        if score < best_score:
            best_score = score
            best_e = e
    print("best e =", best_e, " with score ", best_score)  # noqa
    return best_e


def _tune_lcss(train_X, n_clusters):  # pragma: no cover
    best_e = 0
    best_score = sys.float_info.max
    for e in np.arange(0.0, 0.2, 0.01):
        cls = TimeSeriesKMeans(
            distance="lcss", distance_params={"epsilon": e}, n_clusters=n_clusters
        )
        cls.fit(train_X)
        preds = cls.predict(train_X)
        clusters = len(np.unique(preds))
        if clusters <= 1:
            score = sys.float_info.max
        else:
            score = davies_bouldin_score(train_X, preds)
        print(
            f" Number of clusters ={clusters} epsilon parameter = {e} score = {score}"
        )
        if score < best_score:
            best_score = score
            best_e = e
    print("best e =", best_e, " with score ", best_score)
    return best_e
