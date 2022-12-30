# -*- coding: utf-8 -*-
"""Util functions, refactor."""
__author__ = ["TonyBagnall", "MatthewMiddlehurst"]

import os


def results_present(path, estimator, dataset, res):
    """Check if results are present already."""
    full_path = f"{path}/{estimator}Predictions/{dataset}/testResample{res}.csv"
    full_path2 = f"{path}/{estimator}Predictions/{dataset}/trainResample{res}.csv"
    if os.path.exists(full_path) and os.path.exists(full_path2):
        return True
    return False


def results_present_full_path(path, dataset, res):
    """Duplicate: check if results are present already."""
    full_path = f"{path}/Predictions/{dataset}/testResample{res}.csv"
    full_path2 = f"{path}/Predictions/{dataset}/trainResample{res}.csv"
    if os.path.exists(full_path) and os.path.exists(full_path2):
        return True
    return False
