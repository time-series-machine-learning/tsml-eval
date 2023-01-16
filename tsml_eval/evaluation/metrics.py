# -*- coding: utf-8 -*-
"""Performance metric functions."""

__author__ = ["MatthewMiddlehurst"]

__all__ = ["clustering_accuracy", "davies_bouldin_score_from_file"]

import sys

import numpy as np
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import confusion_matrix, davies_bouldin_score


def clustering_accuracy(y_true, y_pred):
    """Calculate clustering accuracy."""
    matrix = confusion_matrix(y_true, y_pred)
    row, col = linear_sum_assignment(matrix.max() - matrix)
    s = sum([matrix[row[i], col[i]] for i in range(len(row))])
    return s / y_pred.size


def davies_bouldin_score_from_file(X, file_path):
    """Calculate Davies-Bouldin score from a results file."""
    y = np.zeros(len(X))
    with open(file_path, "r") as f:
        lines = f.readlines()
        for i, line in enumerate(lines[3:]):
            y[i] = float(line.split(",")[1])

    clusters = len(np.unique(y))
    if clusters <= 1:
        return sys.float_info.max
    else:
        return davies_bouldin_score(X, y)
