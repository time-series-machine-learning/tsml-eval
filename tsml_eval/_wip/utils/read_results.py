# -*- coding: utf-8 -*-
from csv import reader
from typing import Dict, List


def read_results_from_uea_format(
    path: str,
    meta_col_headers: List[str] = None,
    prediction_col_headers: List[str] = None,
) -> Dict:
    """Read results from uea format.

    Parameters
    ----------
    path: str
        Path to results file csv.
    meta_col_headers: List[str], defaults = None
        Column header for meta data about estimator (third line)
    prediction_col_headers: List[str], defaults = None
        Column header for predictions data (fourth line and onwards)

    Returns
    -------
    dict
        Dict in the following format:
        {
            'first_line_comment': [first line data]
            'estimator_parameters': [second line data]
            'estimator_meta': [third line data]
            'predictions': [forth line and onwards]
        }

    """
    read_dict = {}
    # open file in read mode
    with open(path, "r") as read_obj:
        csv_reader = reader(read_obj)

        read_dict["first_line_comment"] = next(csv_reader)
        read_dict["estimator_parameters"] = next(csv_reader)

        read_dict["estimator_meta"] = []
        if meta_col_headers is not None:
            read_dict["estimator_meta"].append(meta_col_headers)
        read_dict["estimator_meta"].append(next(csv_reader))

        read_dict["predictions"] = []

        if prediction_col_headers is not None:
            read_dict["predictions"].append(prediction_col_headers)

        for row in csv_reader:
            read_dict["predictions"].append(row)

    return read_dict


def read_clusterer_result_from_uea_format(csv_path):
    meta_col_headers = [
        "N/A",
        "build time",
        "test time",
        "N/A",
        "N/A",
        "num classes",
        "num classes",
    ]
    with open(csv_path, "r") as read_obj:
        csv_reader = reader(read_obj)
        next(csv_reader)  # Skip first line
        next(csv_reader)  # Skip second line
        meta = next(csv_reader)  # Skip second line
        num_classes = meta[-1]

    prediction_col_headers = ["True y class", "Predicted y class", "N/A"]

    for i in range(1, int(num_classes) + 1):
        prediction_col_headers.append(f"proba of class {str(i)}")

    return read_results_from_uea_format(
        csv_path, meta_col_headers, prediction_col_headers
    )
