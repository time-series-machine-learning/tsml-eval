"""Utilities for validating results and estimators."""

__all__ = [
    "is_sklearn_estimator",
    "is_sklearn_classifier",
    "is_sklearn_regressor",
    "is_sklearn_clusterer",
    "validate_results_file",
]

from aeon.base import BaseEstimator as AeonBaseEstimator
from sklearn.base import BaseEstimator as SklearnBaseEstimator
from sklearn.base import is_classifier, is_regressor
from tsml.base import BaseTimeSeriesEstimator
from tsml.utils.validation import is_clusterer


def is_sklearn_estimator(estimator):
    """Check if estimator is a scikit-learn estimator."""
    return (
        isinstance(estimator, SklearnBaseEstimator)
        and not isinstance(estimator, AeonBaseEstimator)
        and not isinstance(estimator, BaseTimeSeriesEstimator)
    )


def is_sklearn_classifier(classifier):
    """Check if estimator is a scikit-learn classifier."""
    return is_sklearn_estimator(classifier) and is_classifier(classifier)


def is_sklearn_regressor(regressor):
    """Check if estimator is a scikit-learn regressor."""
    return is_sklearn_estimator(regressor) and is_regressor(regressor)


def is_sklearn_clusterer(clusterer):
    """Check if estimator is a scikit-learn clusterer."""
    return is_sklearn_estimator(clusterer) and is_clusterer(clusterer)


def validate_results_file(file_path):
    """Validate that a results file is in the correct format.

    Validates that the first, second, third and results lines follow the expected
    format. This does not verify that the actual contents of the results file make
    sense.

    Works for classification, regression and clustering results files.

    Parameters
    ----------
    file_path : str
        Path to the results file to be validated, including the file itself.

    Returns
    -------
    valid_file : bool
        True if the results file is valid, False otherwise.
    """
    with open(file_path, "r") as f:
        lines = f.readlines()

    if not _check_first_line(lines[0]) or not _check_second_line(lines[1]):
        return False

    if _check_classification_third_line(lines[2]):
        n_probas = int(lines[2].split(",")[5])
        probabilities = True
    elif _check_clustering_third_line(lines[2]):
        n_probas = int(lines[2].split(",")[6])
        probabilities = True
    elif _check_regression_third_line(lines[2]) or _check_forecasting_third_line(
        lines[2]
    ):
        n_probas = 0
        probabilities = False
    else:
        return False

    for i in range(3, len(lines)):
        if not _check_results_line(
            lines[i], probabilities=probabilities, n_probas=n_probas
        ):
            return False

    return True


def _check_first_line(line):
    line = line.split(",")
    return len(line) >= 5


def _check_second_line(line):
    line = line.split(",")
    return len(line) >= 1


def _check_classification_third_line(line):
    line = line.split(",")
    floats = [0, 1, 2, 3, 4, 5, 7, 8]
    return _check_line_length_and_floats(line, 9, floats)


def _check_regression_third_line(line):
    line = line.split(",")
    floats = [0, 1, 2, 3, 4, 6, 7]
    return _check_line_length_and_floats(line, 8, floats)


def _check_clustering_third_line(line):
    line = line.split(",")
    floats = [0, 1, 2, 3, 4, 5, 6]
    return _check_line_length_and_floats(line, 7, floats)


def _check_forecasting_third_line(line):
    line = line.split(",")
    floats = [0, 1, 2, 3, 4]
    return _check_line_length_and_floats(line, 5, floats)


def _check_line_length_and_floats(line, length, floats):
    if len(line) != length:
        return False

    for i in floats:
        try:
            float(line[i])
        except ValueError:
            return False

    return True


def _check_results_lines(lines, num_results_lines=None, probabilities=True, n_probas=2):
    if num_results_lines is not None:
        assert len(lines) - 3 == num_results_lines

        for i in range(3, num_results_lines):
            assert _check_results_line(
                lines[i], probabilities=probabilities, n_probas=n_probas
            )
    else:
        for i in range(3, 6):
            assert _check_results_line(
                lines[i], probabilities=probabilities, n_probas=n_probas
            )


def _check_results_line(line, probabilities=True, n_probas=2):
    line = line.split(",")

    if len(line) < 2:
        return False

    try:
        float(line[0])
        float(line[1])
    except ValueError:
        return False

    if probabilities:
        if len(line) < 3 + n_probas or line[2] != "":
            return False

        psum = 0
        try:
            for i in range(n_probas):
                psum += float(line[3 + i])
        except ValueError:
            return False

        if psum < 0.999 or psum > 1.001:
            return False
    else:
        n_probas = 0

    if len(line) > 4 + n_probas:
        if line[4 + n_probas] != "":
            return False

        try:
            float(line[5 + n_probas])
        except ValueError:
            return False

    if len(line) > 5 + n_probas and line[5 + n_probas] != "":
        return False

    return True
