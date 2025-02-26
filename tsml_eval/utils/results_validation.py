"""Utilities for validating results."""

__maintainer__ = ["MatthewMiddlehurst"]
__all__ = [
    "validate_results_file",
    "compare_result_file_resample",
]


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
    with open(file_path) as f:
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

        if psum < 0.99999 or psum > 1.00001:
            return False
    else:
        n_probas = 0

    if len(line) > 4 + n_probas:
        if line[4 + n_probas] != "":
            return False

        try:
            float(line[4 + n_probas])
        except ValueError:
            return False

    if len(line) > 5 + n_probas and line[5 + n_probas] != "":
        return False

    return True


def compare_result_file_resample(file_path1, file_path2):
    """Validate that two results files use the same data resample.

    Files are deemed as having the same resample if the file length is the same and all
    true label values are the same in both files.

    Parameters
    ----------
    file_path1 : str
        Path to the first results file to be compared, including the file itself.
    file_path1 : str
        Path to the second results file to be compared, including the file itself.

    Returns
    -------
    same_resample : bool
        True if the results file use the same data resample, False otherwise.
    """
    with open(file_path1) as f:
        lines1 = f.readlines()

    with open(file_path2) as f:
        lines2 = f.readlines()

    if len(lines1) != len(lines2):
        raise ValueError("Input results file have different numbers of lines.")

    for i in range(3, len(lines1)):
        if lines1[i].split(",")[0] != lines2[i].split(",")[0]:
            return False

    return True
