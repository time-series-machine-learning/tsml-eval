"""Utilities for publications."""

__all__ = [
    "extract_publication_csv_from_evaluation",
    "parameter_table_from_estimator_selector",
]


import os
import shutil

import pandas as pd
from aeon.utils.validation._dependencies import _check_soft_dependencies


def extract_publication_csv_from_evaluation(stats, eval_path, write_path):
    """Extract the CSV files from the evaluation directory to a new directory.

    Parameters
    ----------
    stats : str or list of str
        The statistics to extract. The name of a statistic or list of names.
    eval_path : str or list of str
        The path to the evaluation directory generated from the evaluation package.
        If a list, each item is the path to an evaluation directory.
    write_path : str or list of str
        The path to the directory where the CSV files will be written. If a list,
        each item is the path to a directory where the CSV files will be written.
        eval_path and write_path must be the same length.

    Examples
    --------
    >>> from tsml_eval.testing.testing_utils import (
    ...     _TEST_EVAL_PATH, _TEST_OUTPUT_PATH
    ... )
    >>> from tsml_eval.utils.publications import (
    ...     extract_publication_csv_from_evaluation
    ... )
    >>> extract_publication_csv_from_evaluation(
    ...     ["Accuracy", "BalAcc"],
    ...     f"{_TEST_EVAL_PATH}/classification/",
    ...     f"{_TEST_OUTPUT_PATH}/result_files/",
    ... )
    """
    if isinstance(stats, str):
        stats = [stats]

    os.makedirs(write_path, exist_ok=True)

    for stat in stats:
        stat_dir = f"{eval_path}/{stat}/all_resamples/"

        for file in os.listdir(stat_dir):
            if file.endswith(".csv"):
                shutil.copy(f"{stat_dir}/{file}", f"{write_path}/{file}")


def parameter_table_from_estimator_selector(selection_function, estimator_names):
    """Create a table of estimator names and their parameters in LaTeX format.

    Parameters
    ----------
    selection_function : function
        The function that selects the estimator.
    estimator_names : list of str
        The names of the estimators.

    Returns
    -------
    str
        The LaTeX table string.
    """
    parameters = []
    for estimator_name in estimator_names:
        est = selection_function(estimator_name)
        params = est.get_params()
        params.pop("random_state", None)
        params.pop("n_jobs", None)
        params.pop("verbose", None)
        parameters.append(params)

    table = "Estimator & Parameters \\\\ \\hline \n"
    for i, params in enumerate(parameters):
        table += estimator_names[i] + " & "
        for key, value in params.items():
            table += f"{key}: {value}, "
        table += " \\\\ \n"
    return table


def results_table_from_evaluation_csv(
    eval_csv_path: str,
    bold_best: bool = True,
    round_digits: int = 4,
    rank_columns: bool = False,
    higher_is_better: bool = True,
) -> str:
    """Create a table of results from an evaluation CSV file in LaTeX format.

    Parameters
    ----------
    eval_csv_path : str
        Path to the evaluation CSV file.
    bold_best : bool, default=True
        Bold the highest rounded value(s) per column.
    round_digits : int, default=4
        Decimal places for rounding (drives display, best, and ranking).
    rank_columns : bool, default=False
        Append competition rank per column in brackets, e.g. ``0.9123 (1)``.
    higher_is_better : bool, default=True
        Whether higher values are better for determining the best score.

    Returns
    -------
    str
        The LaTeX table string.
    """
    _check_soft_dependencies("jinja2")

    df = pd.read_csv(eval_csv_path)
    df.set_index(df.columns[0], inplace=True)
    df.index.name = None

    def escape_underscores(s):
        if isinstance(s, str):
            return s.replace("_", r"\_")
        return s

    df.index = df.index.map(escape_underscores)
    df.columns = df.columns.map(escape_underscores)

    df = df.round(round_digits)
    best = df.eq(df.max(axis=0) if higher_is_better else df.min(axis=0))
    ranks = df.rank(method="min", ascending=False if higher_is_better else True)

    out = pd.DataFrame(index=df.index, columns=df.columns, dtype="object")
    for c in df.columns:
        for idx, score in df[c].items():
            cell = f"{score}"
            if rank_columns:
                cell += f" ({int(ranks.at[idx, c])})"
            if bold_best and bool(best.at[idx, c]):
                cell = r"\textbf{" + cell + "}"
            out.at[idx, c] = cell

    col_format = "l" + "r" * len(out.columns)
    return out.to_latex(index=True, escape=False, column_format=col_format)
