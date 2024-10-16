"""Quality measures to use with STC."""

import numpy as np
from numba import njit
import numpy as np


@njit(fastmath=True, cache=True)
def binary_information_gain(orderline, c1, c2):
    """Find binary information gain for an order line.

    Parameters
    ----------

    orderline: np.array
        Sorted array of tuples (check).
    c1: int
        Number of cases of the class of interest
    c2: int
        Number of cases of all other classes
    """

    def _binary_entropy(c1, c2):
        ent = 0
        if c1 != 0:
            ent -= c1 / (c1 + c2) * np.log2(c1 / (c1 + c2))
        if c2 != 0:
            ent -= c2 / (c1 + c2) * np.log2(c2 / (c1 + c2))
        return ent

    initial_ent = _binary_entropy(c1, c2)

    total_all = c1 + c2

    bsf_ig = 0
    c1_count = 0
    c2_count = 0

    # evaluate each split point
    for split in range(len(orderline)):
        next_class = orderline[split][1]  # +1 if this class, -1 if other
        if next_class > 0:
            c1_count += 1
        else:
            c2_count += 1

        left_prop = (split + 1) / total_all
        ent_left = _binary_entropy(c1_count, c2_count)

        right_prop = 1 - left_prop
        ent_right = _binary_entropy(
            c1 - c1_count,
            c2 - c2_count,
        )

        ig = initial_ent - left_prop * ent_left - right_prop * ent_right
        bsf_ig = max(ig, bsf_ig)

    return bsf_ig



@njit(fastmath=True, cache=True)
def _moods_median(class0, class1):
    """Calculate Mood's Median test statistic for two treatment levels.

    Mood's median is a non-parameteric test for diffence in medians between
    samples of groups. This version is for two group tests only.

    Parameters
    ----------
    class0: np.ndarray
            Array of distances to the first class.
    class1 (np.array): Array of distances for the second class.

    Returns:
    - float value

    """
    combined = np.concatenate((class0, class1))
    median_value = np.median(
        combined
    )  # np.median is supported in recent Numba releases

    above0 = np.sum(class0 > median_value)
    below0 = len(class0) - above0
    above1 = np.sum(class1 > median_value)
    below1 = len(class1) - above1

    total_above = above0 + above1
    total_below = below0 + below1

    total = total_above + total_below

    expected0_above = total_above * (above0 + below0) / total
    expected0_below = total_below * (above0 + below0) / total
    expected1_above = total_above * (above1 + below1) / total
    expected1_below = total_below * (above1 + below1) / total

    chi_squared_stat = (
        ((above0 - expected0_above) ** 2 / expected0_above)
        + ((below0 - expected0_below) ** 2 / expected0_below)
        + ((above1 - expected1_above) ** 2 / expected1_above)
        + ((below1 - expected1_below) ** 2 / expected1_below)
    )

    return chi_squared_stat


@njit(fastmath=True, cache=True)
def f_stat(class0, class1):
    """
    Calculate the F-statistic for shapelet quality based on two numpy arrays of distances for two classes.
    Parameters:
    - class0 (np.array): Array of distances for the first class.
    - class1 (np.array): Array of distances for the second class.
    Returns:
    - float: The computed F-statistic.
    """

    if len(class0) == 0 or len(class1) == 0:
        return np.inf  # Use NumPy's inf representation

    # Calculate means
    mean_class0 = np.mean(class0)
    mean_class1 = np.mean(class1)
    all_distances = np.concatenate((class0, class1))
    overall_mean = np.mean(all_distances)

    n0 = len(class0)
    n1 = len(class1)
    total_n = n0 + n1

    # Between-class sum of squares
    ssb = (
        n0 * (mean_class0 - overall_mean) ** 2 + n1 * (mean_class1 - overall_mean) ** 2
    )

    # Within-class sum of squares
    ssw = np.sum((class0 - mean_class0) ** 2) + np.sum((class1 - mean_class1) ** 2)

    # Degrees of freedom
    df_between = 1
    df_within = total_n - 2

    # Avoid division by zero
    if df_within <= 0:
        return np.inf

    F_stat = (ssb / df_between) / (ssw / df_within)
    return F_stat


# Kruskal Wallis pre stat uses some methods not compatible with numba
# The Kruskal Wallis is calculated using 2 functions:
# one for the pre_stats values such as unique values, ranks, tie_correction, len(class0), len(class1)..this doesnt invoke the njit as it uses some functions that are incompatible with njit
# another for the actual calculation, compatible with numba, and uses the return values from the pre_stat function

def compute_pre_stats(class0, class1):
    combined_array = np.concatenate((class0, class1))
    ranks = np.argsort(np.argsort(combined_array)) + 1
    unique, counts = np.unique(combined_array, return_counts=True)
    tie_correction = 1 - (
        np.sum(counts**3 - counts) / ((len(combined_array) ** 3) - len(combined_array))
    )
    return ranks, tie_correction, len(class0), len(class1), len(combined_array)


@njit(fastmath=True, cache=True)
def kruskal_wallis_test(ranks, n1, n2, n, tie_correction):
    R1 = np.sum(ranks[:n1])
    R2 = np.sum(ranks[n1:])

    mean_rank1 = R1 / n1
    mean_rank2 = R2 / n2
    mean_rank = np.mean(ranks)

    K = (12 / (n * (n + 1))) * (
        R1 * (mean_rank1 - mean_rank) ** 2 + R2 * (mean_rank2 - mean_rank) ** 2
    )

    K /= tie_correction
    return K
