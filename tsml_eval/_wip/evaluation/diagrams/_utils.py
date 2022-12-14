# -*- coding: utf-8 -*-
import warnings
from typing import List, Union

import numpy as np
import pandas as pd

from tsml_eval.evaluation import MetricResults, metric_result_to_summary


def metric_result_to_df(
    result: Union[pd.DataFrame, List[MetricResults]]
) -> pd.DataFrame:
    """Converts and validates dataframe in correct format.

    Parameters
    ----------
    result: Union[pd.DataFrame, List[MetricResults]]
        Dataframe or list of metric results.

    Returns
    -------
    pd.DataFrame
        Dataframe in correct format of:
        ----------------------------------
        | estimator | dataset | metric1  | metric2 |
        | cls1      | data1   | 1.2      | 1.2     |
        | cls2      | data2   | 3.4      | 1.4     |
        | cls1      | data2   | 1.4      | 1.3     |
        | cls2      | data1   | 1.3      | 1.2     |
        ----------------------------------
    """
    df = result
    if not isinstance(result, pd.DataFrame):
        df = metric_result_to_summary(result)

    df = _check_df(df)
    return df


def _check_df(df: pd.DataFrame) -> pd.DataFrame:
    """Check if data frame is valid.

    Parameters
    ----------
    df: pd.DataFrame
        Data frame to check.

    Returns
    -------
    pd.DataFrame
        Validated dataframe
    """
    datasets = set((df.iloc[:, 1]).unique())
    estimators = (df.iloc[:, 0]).unique()

    remove_datasets = set()

    for estimator in estimators:
        curr = df[df.iloc[:, 0] == estimator]
        curr_datasets = set((curr.iloc[:, 1]).unique())
        remove_datasets = remove_datasets.union(datasets.difference(curr_datasets))

    df = df[~df.iloc[:, 1].isin(remove_datasets)]
    datasets = set((df.iloc[:, 1]).unique())

    num_datasets = len(datasets)
    for estimator in estimators:
        curr = df[df.iloc[:, 0] == estimator]
        if len(curr.iloc[:, 1]) != num_datasets:
            warnings.warn(
                f"Number of datasets for estimator {estimator} is not equal to "
                f"number of datasets in dataframe. Removing {estimator} from dataframe."
            )
            test1 = curr.iloc[:, 1]
            df = df[df.iloc[:, 0] != estimator]
    return df
