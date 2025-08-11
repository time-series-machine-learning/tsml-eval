"""Tests for quality measures. Work out some test examples and check code generates
them here."""

import sys
import os

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(parent_dir)


# sys.path.append("tsml_eval/_wip/transformations/collection/shapelet_based")

from tsml_eval._wip.transformations.collection.shapelet_based import (
    _quality_measures as qm,
)


import numpy as np


class0 = np.array([0.5, 1.5, 2.2])
class1 = np.array([1.2, 1.8, 2.5, 3.1, 3.5])


# Testing F-statistics
f_statistic = qm.f_stat(class0, class1)
print(f"F-statistic: {f_statistic}")



# Testing Mood's Median
chi_statistic = qm._moods_median(class0, class1)
print(f"Mood's Median chi-square statistic: {chi_statistic}")



# Kruskal Wallis


# LOW DIFF PAIR
data1_ = np.array([1, 3, 5, 7, 9])
data2_ = np.array([2, 4, 6, 8, 12])
# data1_ = np.array([])
# data2_ = np.array([])
ranks, tie_correction, n1, n2, n = qm.compute_pre_stats(data1_, data2_)
kw_statistic = qm.kruskal_wallis_test(ranks, n1, n2, n, tie_correction)
print(f"Kruskal-Wallis statistic for low diff pair of arrays: {kw_statistic}")


# The tester may test the function with the following pair where the Difference in the pairs is greater
# uncomment the 5 lines of code below

# HIGH DIFF PAIR
# data1 = np.array([1, 1, 2, 2, 3]) #uncomment
# data2 = np.array([5, 5, 6, 6, 7]) #uncomment
# kw_statistic = qm.kruskal_wallis_test(ranks, n1, n2, n, tie_correction) #uncomment
# print(f"Kruskal-Wallis statistic for high diff pair of arrays: {kw_statistic}") #uncomment
# /Users/abhashshrestha/Documents/test_clone/tsml-eval/tsml_eval/_wip/transformations/collection/shapelet_based/tests/test_quality_measures.py


def estimate_parameters(data):
    """Estimate the mean and covariance matrix of a dataset."""
    mu = np.mean(data, axis=0)  # Ensure mean is computed over an axis for array result
    centered_data = data - mu  # Broadcasting subtraction to center data
    covariance = np.cov(centered_data, rowvar=False)
    return mu, np.atleast_2d(covariance)  # Ensure covariance is 2D


mu1, Sigma1 = estimate_parameters(data1_)
mu2, Sigma2 = estimate_parameters(data2_)
wass_dist = qm.wasserstein_distance_gaussian(
    mu1.flatten(), Sigma1, mu2.flatten(), Sigma2
)
print("Wasserstein Distance for Estimated Gaussian Distributions:", wass_dist)


emp_wd = qm.wasserstein_distance_empirical(data1_, data2_)
print("Wasserstein Distance for Empirical Distributions:", emp_wd)


ks = qm.kolmogorov_test(data1_, data2_)
print("Kolmogorov Smirnov for Empirical Distributions:", ks)
