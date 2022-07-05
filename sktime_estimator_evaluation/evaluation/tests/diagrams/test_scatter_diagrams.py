# -*- coding: utf-8 -*-
"""Test for scatter diagrams."""
import pandas as pd

from sktime_estimator_evaluation.evaluation.diagrams import create_scatter_diagram

DEBUG = True


def test_scatter_diagrams():
    df = pd.DataFrame({
        'estimator': ['cls1', 'cls2', 'cls3', 'cls1', 'cls2', 'cls3', 'cls1', 'cls2', 'cls3'],
        'dataset': ['data1', 'data1', 'data1','data2', 'data2', 'data2', 'data3', 'data3', 'data3'],
        'metric1': [0.9, 0.1, 0.2, 0.3, 0.44, 0.86, 0.92, 0.2, 0.122],
        'metric2': [0.2, 0.3, 0.4, 0.2, 0.3, 0.4, 0.2, 0.3, 0.4],
    })

    figures = create_scatter_diagram(
        df,
        compare_estimator_columns=['cls1', 'cls2', 'cls3'],
        # output_path='/Users/chris/Documents/Projects/sktime-estimator-evaluation/sktime_estimator_evaluation/evaluation/tests/diagrams/out',
        # compare_metric_columns=['metric1'],
    )

    for fig in figures:
        fig.show()

    joe = ''
