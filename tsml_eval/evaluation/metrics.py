"""Performance metric functions."""

__author__ = ["MatthewMiddlehurst"]

__all__ = ["clustering_accuracy_score"]


from scipy.optimize import linear_sum_assignment
from sklearn.metrics import confusion_matrix


def clustering_accuracy_score(y_true, y_pred):
    """Calculate clustering accuracy."""
    matrix = confusion_matrix(y_true, y_pred)
    row, col = linear_sum_assignment(matrix.max() - matrix)
    s = sum([matrix[row[i], col[i]] for i in range(len(row))])
    return s / y_pred.size
