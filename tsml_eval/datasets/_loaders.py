"""Data loading functions."""

import os

import numpy as np

__all__ = [
    "load_minimal_chinatown",
    "load_unequal_minimal_chinatown",
    "load_equal_minimal_japanese_vowels",
    "load_minimal_japanese_vowels",
    "load_minimal_gas_prices",
    "load_unequal_minimal_gas_prices",
    "load_minimal_cardano_sentiment",
    "load_unequal_minimal_cardano_sentiment",
]

from aeon.datasets import load_from_ts_file


def load_minimal_chinatown(
    split: None | str = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Load MinimalChinatown time series classification problem.

    This is an equal length univariate time series classification problem. It is a
    stripped down version of the ChinaTown problem that is used in correctness tests
    for classification. It loads a two class classification problem with 20 cases
    for both the train and test split and a series length of 24.

    For the full dataset see
    http://timeseriesclassification.com/description.php?Dataset=Chinatown

    Parameters
    ----------
    split: "TRAIN", "TEST" or None, default=None
        Whether to load the train or test instances of the problem. If None, loads
        both train and test instances (in a single container).

    Returns
    -------
    X: np.ndarray
        The time series data for the problem of shape (20,1,24).
    y: np.ndarray
        The class labels for each case in X.

    Examples
    --------
    >>> from tsml_eval.datasets._loaders import load_minimal_chinatown
    >>> X, y = load_minimal_chinatown()
    """
    return _load_provided_dataset("MinimalChinatown", split)


def load_unequal_minimal_chinatown(
    split: None | str = None,
) -> tuple[list[np.ndarray], np.ndarray]:
    """Load UnequalMinimalChinatown time series classification problem.

    This is an unequal length univariate time series classification problem. It is a
    stripped down version of the ChinaTown problem that is used in correctness tests
    for classification. Parts of the original series have been randomly removed. It
    loads a two class classification problem with 20 cases for both the train and test
    split.

    For the full dataset see
    http://timeseriesclassification.com/description.php?Dataset=Chinatown

    Parameters
    ----------
    split: "TRAIN", "TEST" or None, default=None
        Whether to load the train or test instances of the problem. If None, loads
        both train and test instances (in a single container).

    Returns
    -------
    X: list of np.ndarray
        The time series data for the problem in a list of size 20 containing 2D
        ndarrays.
    y: np.ndarray
        The class labels for each case in X.

    Examples
    --------
    >>> from tsml_eval.datasets._loaders import load_unequal_minimal_chinatown
    >>> X, y = load_unequal_minimal_chinatown()
    """
    return _load_provided_dataset("UnequalMinimalChinatown", split)


def load_equal_minimal_japanese_vowels(
    split: None | str = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Load the EqualMinimalJapaneseVowels time series classification problem.

    This is an equal length multivariate time series classification problem. It is a
    stripped down version of the JapaneseVowels problem that is used in correctness
    tests for classification. It has been altered so all series are equal length. It
    loads a nine class classification problem with 20 cases for both the train and test
    split, 12 channels and a series length of 25.

    For the full dataset see
    http://www.timeseriesclassification.com/description.php?Dataset=JapaneseVowels

    Parameters
    ----------
    split: "TRAIN", "TEST" or None, default=None
        Whether to load the train or test instances of the problem. If None, loads
        both train and test instances (in a single container).

    Returns
    -------
    X: np.ndarray
        The time series data for the problem of shape (20,12,25).
    y: np.ndarray
        The class labels for each case in X.

    Examples
    --------
    >>> from tsml_eval.datasets._loaders import load_equal_minimal_japanese_vowels
    >>> X, y = load_equal_minimal_japanese_vowels()
    """
    return _load_provided_dataset("EqualMinimalJapaneseVowels", split)


def load_minimal_japanese_vowels(
    split: None | str = None,
) -> tuple[list[np.ndarray], np.ndarray]:
    """Load the MinimalJapaneseVowels time series classification problem.

    This is an unequal length multivariate time series classification problem. It is a
    stripped down version of the JapaneseVowels problem that is used in correctness
    tests for classification. It loads a nine class classification problem with 20 cases
    for both the train and test split and 12 channels.

    For the full dataset see
    http://www.timeseriesclassification.com/description.php?Dataset=JapaneseVowels

    Parameters
    ----------
    split: "TRAIN", "TEST" or None, default=None
        Whether to load the train or test instances of the problem. If None, loads
        both train and test instances (in a single container).

    Returns
    -------
    X: list of np.ndarray
        The time series data for the problem in a list of size 20 containing 2D
        ndarrays.
    y: np.ndarray
        The class labels for each case in X.

    Examples
    --------
    >>> from tsml_eval.datasets._loaders import load_minimal_japanese_vowels
    >>> X, y = load_minimal_japanese_vowels()
    """
    return _load_provided_dataset("MinimalJapaneseVowels", split)


def load_minimal_gas_prices(
    split: None | str = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Load the MinimalGasPrices time series extrinsic regression problem.

    This is an equal length univariate time series regression problem. It is a
    stripped down version of the GasPricesSentiment problem that is used in correctness
    tests for regression. It loads a regression problem with 20 cases for both the train
    and test split and a series length of 20.

    Parameters
    ----------
    split: "TRAIN", "TEST" or None, default=None
        Whether to load the train or test instances of the problem. If None, loads
        both train and test instances (in a single container).

    Returns
    -------
    X: np.ndarray
        The time series data for the problem of shape (20,1,20).
    y: np.ndarray
        The labels for each case in X.

    Examples
    --------
    >>> from tsml_eval.datasets._loaders import load_minimal_gas_prices
    >>> X, y = load_minimal_gas_prices()
    """
    return _load_provided_dataset("MinimalGasPrices", split)


def load_unequal_minimal_gas_prices(
    split: None | str = None,
) -> tuple[list[np.ndarray], np.ndarray]:
    """Load the UnequalMinimalGasPrices time series extrinsic regression problem.

    This is an unequal length univariate time series regression problem. It is a
    stripped down version of the GasPricesSentiment problem that is used in correctness
    tests for regression. Parts of the original series have been randomly removed. It
    loads a regression problem with 20 cases for both the train and test split.

    Parameters
    ----------
    split: "TRAIN", "TEST" or None, default=None
        Whether to load the train or test instances of the problem. If None, loads
        both train and test instances (in a single container).

    Returns
    -------
    X: list of np.ndarray
        The time series data for the problem in a list of size 20 containing 2D
        ndarrays.
    y: np.ndarray
        The labels for each case in X.

    Examples
    --------
    >>> from tsml_eval.datasets._loaders import load_unequal_minimal_gas_prices
    >>> X, y = load_unequal_minimal_gas_prices()
    """
    return _load_provided_dataset("UnequalMinimalGasPrices", split)


def load_minimal_cardano_sentiment(
    split: None | str = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Load the MinimalCardanoSentiment time series extrinsic regression problem.

    This is an equal length multivariate time series regression problem. It is a
    stripped down version of the CardanoSentiment problem that is used in correctness
    tests for regression. It loads a regression problem with 20 cases for both the train
    and test split and a series length of 24.

    Parameters
    ----------
    split: "TRAIN", "TEST" or None, default=None
        Whether to load the train or test instances of the problem. If None, loads
        both train and test instances (in a single container).

    Returns
    -------
    X: np.ndarray
        The time series data for the problem of shape (20,2,24).
    y: np.ndarray
        The labels for each case in X.

    Examples
    --------
    >>> from tsml_eval.datasets._loaders import load_minimal_cardano_sentiment
    >>> X, y = load_minimal_cardano_sentiment()
    """
    return _load_provided_dataset("MinimalCardanoSentiment", split)


def load_unequal_minimal_cardano_sentiment(
    split: None | str = None,
) -> tuple[list[np.ndarray], np.ndarray]:
    """Load the MinimalCardanoSentiment time series extrinsic regression problem.

    This is an unequal length multivariate time series regression problem. It is a
    stripped down version of the CardanoSentiment problem that is used in correctness
    tests for regression. Parts of the original series have been randomly removed. It
    loads a regression problem with 20 cases for both the train and test split.

    Parameters
    ----------
    split: "TRAIN", "TEST" or None, default=None
        Whether to load the train or test instances of the problem. If None, loads
        both train and test instances (in a single container).

    Returns
    -------
    X: list of np.ndarray
        The time series data for the problem in a list of size 20 containing 2D
        ndarrays.
    y: np.ndarray
        The labels for each case in X.

    Examples
    --------
    >>> from tsml_eval.datasets._loaders import load_unequal_minimal_cardano_sentiment
    >>> X, y = load_unequal_minimal_cardano_sentiment()
    """
    return _load_provided_dataset("UnequalMinimalCardanoSentiment", split)


def _load_provided_dataset(
    name: str,
    split: None | str = None,
):
    """Load baked in time series datasets.

    Loads data from the provided tsml dataset files only.

    Parameters
    ----------
    name : str
        File name to load from.
    split: "TRAIN", "TEST" or None, default=None
        Whether to load the train or test instances of the problem. If None, loads
        both train and test instances (in a single container).

    Returns
    -------
    X: np.ndarray or list of np.ndarray
        The time series data for the problem in a 3D array if the data is equal length
        or a list containing 2D arrays if it is unequal.
    y: np.ndarray
        The labels for each case in X.
    """
    if isinstance(split, str):
        split = split.upper()

    if split in ("TRAIN", "TEST"):
        fname = name + "_" + split + ".ts"
        path = os.path.join(os.path.dirname(__file__), name, fname)
        X, y = load_from_ts_file(path)
    # if split is None, load both train and test set
    elif split is None:
        fname = name + "_TRAIN.ts"
        path = os.path.join(os.path.dirname(__file__), name, fname)
        X_train, y_train = load_from_ts_file(path)

        fname = name + "_TEST.ts"
        path = os.path.join(os.path.dirname(__file__), name, fname)
        X_test, y_test = load_from_ts_file(path)

        X = (
            X_train + X_test
            if isinstance(X_train, list)
            else np.concatenate([X_train, X_test])
        )
        y = np.concatenate([y_train, y_test])
    else:
        raise ValueError("Invalid `split` value =", split)

    return X, y
