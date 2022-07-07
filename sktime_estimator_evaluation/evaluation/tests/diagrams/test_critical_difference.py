# -*- coding: utf-8 -*-
"""Test for critical difference diagram."""
import random
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sktime_estimator_evaluation.evaluation.diagrams import create_critical_difference_diagram

def test_critical_difference_from_csv():
    # df = pd.read_csv(os.path.abspath("../test_results/train/all/all.csv"))
    df = pd.read_csv(os.path.abspath("../test_results/test.csv"))
    temp = df.columns[2:]
    df = df[['estimator_name', 'dataset', *temp]]
    fig = create_critical_difference_diagram(df)
    estimators = df['estimator_name'].unique()

    for val in fig:
        val.show()
    joe = ''
