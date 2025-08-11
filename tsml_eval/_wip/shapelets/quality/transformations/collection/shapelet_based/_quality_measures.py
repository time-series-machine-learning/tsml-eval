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
    if n1 == 0 or n2 == 0:
        return -1  # Return a default value indicating an invalid test scenario

    R1 = np.sum(ranks[:n1])
    R2 = np.sum(ranks[n1:])

    mean_rank1 = R1 / n1
    mean_rank2 = R2 / n2
    mean_rank = np.mean(ranks)

    K = (12 / (n * (n + 1))) * (
        R1 * (mean_rank1 - mean_rank) ** 2 + R2 * (mean_rank2 - mean_rank) ** 2
    )
    if tie_correction != 0:
        K /= tie_correction
    else:
        return -1  # Return a default value if tie_correction is zero

    return K


@njit(nopython=True)
def matrix_sqrt(mat):
    """Compute the square root of a matrix using eigenvalue decomposition."""
    eigenvalues, eigenvectors = np.linalg.eigh(mat)
    sqrt_eigenvalues = np.sqrt(eigenvalues)
    return eigenvectors @ np.diag(sqrt_eigenvalues) @ eigenvectors.T


@njit(nopython=True)
def squared_euclidean_distance(vec1, vec2):
    """Calculate the squared Euclidean distance between two vectors."""
    result = 0.0
    for i in range(vec1.size):  # Using .size to safely handle dimensionality
        result += (vec1[i] - vec2[i]) ** 2
    return result


def estimate_parameters(data):
    """Estimate the mean and covariance matrix of a dataset."""
    mu = np.mean(data, axis=0)  # Ensure mean is computed over an axis for array result
    centered_data = data - mu  # Broadcasting subtraction to center data
    covariance = np.cov(centered_data, rowvar=False)
    return mu, np.atleast_2d(covariance)  # Ensure covariance is 2D


@njit(fastmath=True, cache=True)
def wasserstein_distance_gaussian(mu1, Sigma1, mu2, Sigma2):
    """Compute the Wasserstein distance between two Gaussian distributions."""
    mean_diff = squared_euclidean_distance(mu1, mu2)
    trace_Sigma1 = np.trace(Sigma1)
    trace_Sigma2 = np.trace(Sigma2)
    root_Sigma1 = matrix_sqrt(Sigma1)
    product_matrix = np.dot(root_Sigma1, np.dot(Sigma2, root_Sigma1))
    root_product = matrix_sqrt(product_matrix)
    trace_root_product = np.trace(root_product)
    B_squared = trace_Sigma1 + trace_Sigma2 - 2 * trace_root_product
    Wasserstein_dist_squared = mean_diff + B_squared
    return np.sqrt(Wasserstein_dist_squared)


#     return float(w_distance)
@njit(fastmath=True, cache=True)
def wasserstein_distance_empirical(dist1, dist2):
    # Convert lists to numpy arrays if they are not already (Numba requires explicit array type)
    dist1 = np.asarray(dist1, dtype=np.float64)
    dist2 = np.asarray(dist2, dtype=np.float64)

    # Check which list is shorter and pad it with its mean value to match the lengths
    if len(dist1) < len(dist2):
        pad_length = len(dist2) - len(dist1)
        padding = np.full(pad_length, np.mean(dist1))
        dist1 = np.concatenate((dist1, padding))
    elif len(dist2) < len(dist1):
        pad_length = len(dist1) - len(dist2)
        padding = np.full(pad_length, np.mean(dist2))
        dist2 = np.concatenate((dist2, padding))

    # Sort both lists
    sorted_dist1 = np.sort(dist1)
    sorted_dist2 = np.sort(dist2)

    # Calculate the average of the absolute differences
    total_diff = 0.0
    for i in range(len(sorted_dist1)):
        total_diff += np.abs(sorted_dist1[i] - sorted_dist2[i])

    # Calculate mean by dividing the total_diff by the number of elements
    w_distance = total_diff / len(sorted_dist1)

    return float(w_distance)


@njit(fastmath=True, cache=True)
def kolmogorov_test(distance1, distance2):

    # Combine and sort the data
    all_data = np.sort(np.concatenate((distance1, distance2)))
    n1 = len(distance1)
    n2 = len(distance2)

    # Compute EDFs
    edf1 = np.array([np.sum(distance1 <= x) / n1 for x in all_data])
    edf2 = np.array([np.sum(distance2 <= x) / n2 for x in all_data])

    # Calculate KS Statistic
    ks_statistic = np.max(np.abs(edf1 - edf2))

    return ks_statistic
