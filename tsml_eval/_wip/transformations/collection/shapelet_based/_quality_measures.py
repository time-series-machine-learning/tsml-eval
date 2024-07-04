"""Quality measures to use with STC."""

import numpy as np
from numba import njit


@njit(fastmath=True, cache=True)
def binary_information_gain(orderline, c1, c2):
    """Find binary information gain for an order line.

    Parameters
    ----------

    orderline: 2D np.array (?) of what?
        Orderline for a given shapelet
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


# def calculate_f_statistic(orderline):
#     """
#     Calculate the F-statistic for shapelet quality based on a list of tuples containing distances and class labels.

#     Parameters:
#     - orderline (list of tuples): Each tuple contains (distance, class_label).

#     Returns:
#     - float: The computed F-statistic.
#     """

#     class_distances = {}
#     for distance, label in orderline:
#         if label not in class_distances:
#             class_distances[label] = []
#         class_distances[label].append(distance)

#     #F-stat Calc
#     class_means = {cls: np.mean(dists) for cls, dists in class_distances.items()} # each key = class label,  each value = mean distance for that class
#     all_distances = [dist for dists in class_distances.values() for dist in dists]
#     overall_mean = np.mean(all_distances)
#     n = len(all_distances)  # Total number of distance measurements
#     C = len(class_distances)  # Number of classes


#     between_class_sum_of_squares = sum(len(dists) * (class_mean - overall_mean) ** 2
#                                        for cls, class_mean in class_means.items())
#     within_class_sum_of_squares = sum((distance - class_means[label]) ** 2
#                                       for distance, label in orderline)

#     # degrees of freedom
#     df_between = C - 1
#     df_within = n - C

#     #  F-stat
#     if df_within == 0:
#         return float('inf')
#     F_stat = (between_class_sum_of_squares / df_between) / (within_class_sum_of_squares / df_within)

#     return F_stat


@njit(fastmath=True, cache=True)
def calculate_moods_median(class0, class1):
    """
    calculate Mood's Median test statistic

    Parameters:
    - class0 (np.array): Array of distances for the first class.
    - class1 (np.array): Array of distances for the second class.

    Returns:
    - float value
    """
    combined = np.concatenate([class0, class1])
    median_value = np.median(combined)

    above0 = np.sum(class0 > median_value)
    below0 = len(class0) - above0
    above1 = np.sum(class1 > median_value)
    below1 = len(class1) - above1

    contingency_table = np.array([[above0, below0], [above1, below1]])

    total_above = contingency_table[:, 0].sum()
    total_below = contingency_table[:, 1].sum()

    total = total_above + total_below
    expected = np.array(
        [
            [
                total_above * (above0 + below0) / total,
                total_below * (above0 + below0) / total,
            ],
            [
                total_above * (above1 + below1) / total,
                total_below * (above1 + below1) / total,
            ],
        ]
    )

    # Compute the chi-square statistic
    chi_squared_stat = np.sum((contingency_table - expected) ** 2 / expected)

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
    # means
    mean_class0 = np.mean(class0)
    mean_class1 = np.mean(class1)
    all_distances = np.concatenate([class0, class1])
    overall_mean = np.mean(all_distances)

    n0 = len(class0)
    n1 = len(class1)
    total_n = n0 + n1  # Total no. of dist measurements

    # between-class sum of squares
    between_class_sum_of_squares = (
        n0 * (mean_class0 - overall_mean) ** 2 + n1 * (mean_class1 - overall_mean) ** 2
    )

    # within-class sum of squares
    within_class_sum_of_squares = np.sum((class0 - mean_class0) ** 2) + np.sum(
        (class1 - mean_class1) ** 2
    )

    # Degrees of freedom
    df_between = 1  # Number of classes - 1 (2 classes - 1)
    df_within = total_n - 2  # Total number of observations - number of classes

    # Calculate F-statistic
    if df_within == 0:  # Avoid division by zero
        return float("inf")
    F_stat = (between_class_sum_of_squares / df_between) / (
        within_class_sum_of_squares / df_within
    )

    return F_stat


# Example usage
class0 = np.array([0.5, 1.5, 2.2])
class1 = np.array([1.2, 1.8, 2.5])

f_statistic = calculate_f_statistic(class0, class1)
print(f"F-statistic: {f_statistic}")
