# -*- coding: utf-8 -*-
import itertools
import os.path
from typing import List, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from tsml_eval._wip.evaluation.diagrams._utils import metric_result_to_df
from tsml_eval.evaluation import MetricResults


def scatter_diagram(
    metric_results: Union[pd.DataFrame, List[MetricResults]],
    output_path=None,
    compare_estimators_from: List[str] = None,
    compare_estimators_to: List[str] = None,
    compare_dataset_columns: List[str] = None,
    compare_metric_columns: List[str] = None,
    top_half_color: str = "turquoise",
    top_half_alpha: float = 0.5,
    bottom_half_color: str = "white",
    bottom_half_alpha: float = 0.0,
    figure_width: float = 5,
    figure_height: float = 5,
    label_font_size: float = 10,
    label_x: float = 0.2,
    label_y: float = 0.8,
) -> Union[plt.Figure, List[plt.Figure]]:
    """Create a critical difference diagram.

    Parameters
    ----------
    metric_results: pd.DataFrame or List[MetricResults]
        If a List[MetricResults] is passed, then it is formatted to correct DF. If a
        data frame is passed it should have three columns index 0 should be the
        estimator names, index 1 should be the dataset and index 3 and onwards should
        be the estimators metric scored for the datasets. For examples:
        ----------------------------------
        | estimator | dataset | metric1  | metric2 |
        | cls1      | data1   | 1.2      | 1.2     |
        | cls2      | data2   | 3.4      | 1.4     |
        | cls1      | data2   | 1.4      | 1.3     |
        | cls2      | data1   | 1.3      | 1.2     |
        ----------------------------------
    output_path: str, defaults = None
        String that is the path to output the figure. If not specified then figure
        isn't written
    compare_estimators_from: List[str], defaults = None
        List of strings that specify which estimators you want to compare from. If
        left as None, all estimators are compared from.
    compare_estimators_to: List[str], defaults = None
        List of strings that specify which estimators you want to compare to. If
        left as None, all estimators are compared to.
    compare_dataset_columns: List[str], defaults = None
        List of strings that specify which datasets you want to compare to. If
        left as None, all datasets are compared.
    compare_metric_columns: List[str], defaults = None
        List of strings that specify which metrics you want to compare to. If
        left as None, all metrics are compared.
    top_half_color: str, defaults = '0.0'
        The matplotlib color of the top half shaded in.
    top_half_alpha: float, defaults = 0.5
        The alpha value of the top half shaded in.
    bottom_half_color: str, defaults = '0.0'
        The matplotlib color of the bottom half shaded in.
    bottom_half_alpha: float, defaults = 0.5
        The alpha value of the bottom half shaded in.
    figure_width: float, defaults = None
        Width of the figure. If not set then will be automatically defined.
    figure_height: float, defaults = None
        Height of figure. If not set then will be automatically defined.
    label_font_size: float, defaults = 15
        Fontsize for labels of graphic.
    label_y: float, defaults = 0.8
        Y-coordinate for labels of graphic.
    label_x: float, defaults = 0.2
        X-coordinate for labels of graphic.

    Returns
    -------
    plt.Figure or List[plt.Figure]
        If more than one metric passed then a list of critical difference diagram
        figures is return else plt.Figure is returned.
    """
    df = metric_result_to_df(metric_results)
    all_estimator = list(set(df["estimator"]))
    if compare_estimators_from is None:
        compare_estimators_from = all_estimator

    if compare_estimators_to is None:
        compare_estimators_to = all_estimator

    if compare_dataset_columns is None:
        compare_dataset_columns = list(set(df["dataset"]))

    if compare_metric_columns is None:
        compare_metric_columns = list(set(df.columns[2:]))

    if output_path is not None:
        if not os.path.isdir(output_path):
            raise ValueError("Output path must be a directory")

    figures = []

    for comb in itertools.product(compare_estimators_from, compare_estimators_to):
        if comb[0] == comb[1]:
            continue
        curr_compare_to_scores = df[df["estimator"] == comb[0]]
        curr_compare_against_scores = df[df["estimator"] == comb[1]]
        compare_df = pd.concat([curr_compare_to_scores, curr_compare_against_scores])
        result = _plot_scatter_diagram(
            compare_df,
            output_path=output_path,
            metrics=compare_metric_columns,
            datasets=compare_dataset_columns,
            top_half_color=top_half_color,
            top_half_alpha=top_half_alpha,
            bottom_half_color=bottom_half_color,
            bottom_half_alpha=bottom_half_alpha,
            label_x=label_x,
            label_y=label_y,
            label_font_size=label_font_size,
            figure_width=figure_width,
            figure_height=figure_height,
        )

        figures = [*figures, *result]

    return figures


def _plot_scatter_diagram(
    df: pd.DataFrame,
    output_path: Union[str, None],
    metrics: List[str],
    datasets: List[str],
    top_half_color: str,
    top_half_alpha: float,
    bottom_half_color: str,
    bottom_half_alpha: float,
    label_x: float,
    label_y: float,
    label_font_size: float,
    figure_width: float,
    figure_height: float,
):
    """Create scatter plot for two classifiers.

    Parameters
    ----------
    df: pd.DataFrame
        The data frame should have three columns index 0 should be the estimators
        names, index 1 should be the dataset and index 3 and onwards should be the
         estimators metric scored for the datasets. For examples:
        ----------------------------------
        | estimator | dataset | metric1  | metric2 |
        | cls1      | data1   | 1.2      | 1.2     |
        | cls2      | data2   | 3.4      | 1.4     |
        | cls1      | data2   | 1.4      | 1.3     |
        | cls2      | data1   | 1.3      | 1.2     |
        ----------------------------------
        There should only be two estimators in the dataframe.
    output_path: str
        String that is the path to output the figure. If not specified then figure
        isn't written
    metrics: List[str]
        List of strings that specify which metrics you want to compare to. If
        left as None, all metrics are compared.
    datasets: List[str]
        List of strings that specify which datasets you want to compare to. If
        left as None, all datasets are compared.
    top_half_color: str
        The matplotlib color of the top half shaded in.
    top_half_alpha: float
        The alpha value of the top half shaded in.
    bottom_half_color: str
        The matplotlib color of the bottom half shaded in.
    bottom_half_alpha: float
        The alpha value of the bottom half shaded in.
    figure_width: float
        Width of the figure. If not set then will be automatically defined.
    figure_height: float
        Height of figure. If not set then will be automatically defined.
    label_font_size: float
        Fontsize for labels of graphic.
    label_y: float
        Y-coordinate for labels of graphic.
    label_x: float
        X-coordinate for labels of graphic.
    """
    figures = []
    estimators = list(set(df["estimator"]))
    metric_score_dict = {}
    for dataset in datasets:
        curr_dataset_df = df[df["dataset"] == dataset]
        for metric in metrics:
            if metric not in metric_score_dict:
                metric_score_dict[metric] = [[], []]
            curr_metric_df = curr_dataset_df[["estimator", metric]]

            index_zero = float(
                curr_metric_df[curr_metric_df["estimator"] == estimators[0]][metric]
            )

            index_one = float(
                curr_metric_df[curr_metric_df["estimator"] == estimators[1]][metric]
            )

            metric_score_dict[metric][0].append(index_zero)
            metric_score_dict[metric][1].append(index_one)

    for metric in metric_score_dict:
        curr_dict = metric_score_dict[metric]
        x = curr_dict[0]
        y = curr_dict[1]

        zeros = np.zeros(3)
        zeros[-1] = 1
        middle_line = list(zeros)

        fig, ax = plt.subplots(1, 1, figsize=(figure_width, figure_height))
        ax.plot(x, y, "k.")

        # bottom triangle fill
        ax.fill_between(
            middle_line, middle_line, color=bottom_half_color, alpha=bottom_half_alpha
        )
        # top triangle fill
        ax.fill_betweenx(
            middle_line, middle_line, color=top_half_color, alpha=top_half_alpha
        )

        plt.xlim(0, 1)
        plt.ylim(0, 1)
        plt.xlabel(f"{estimators[0]} {metric}")
        plt.ylabel(f"{estimators[1]} {metric}")
        ax.text(
            label_y,
            label_x,
            f"{estimators[0]} \nis better here",
            ha="center",
            fontsize=label_font_size,
        )
        ax.text(
            label_x,
            label_y,
            f"{estimators[1]} \nis better here",
            ha="center",
            fontsize=label_font_size,
        )

        fig.tight_layout()
        figures.append(fig)
        if output_path is not None:
            if not os.path.isdir(f"{output_path}/{metric}"):
                os.makedirs(f"{output_path}/{metric}")
            fig.savefig(f"{output_path}/{metric}/{estimators[0]}-{estimators[1]}.png")
    return figures
