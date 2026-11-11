# Extracting Features from Random Subseries: A Hybrid Pipeline for Time Series Classification and Extrinsic Regression

Matthew Middlehurst and Anthony Bagnall. International Workshop on Advanced Analytics
and Learning on Temporal Data (AALTD), 2023.
[Paper](https://doi.org/10.1007/978-3-031-49896-1_8)

The paper introduces RIST, a pipeline which combines features extracted from random
intervals and random dilated shapelets, and evaluates it for both time series
classification and extrinsic regression.

![Critical difference diagram of RMSE ranks for RIST and five other regressors](https://raw.githubusercontent.com/time-series-machine-learning/rist-pipeline-2023/main/figures/regression_cd.png)

RMSE critical difference diagram comparing RIST with five regression algorithms,
using the average over 5 resamples on 55 regression datasets (Figure 6 in the paper).

The code and results for the paper are in the
[rist-pipeline-2023](https://github.com/time-series-machine-learning/rist-pipeline-2023)
repository:

- [Results](https://github.com/time-series-machine-learning/rist-pipeline-2023/tree/main/results)
  for the 112 UCR classification datasets and 55 regression datasets
- [Datasets](https://github.com/time-series-machine-learning/rist-pipeline-2023#datasets)
  used in the experiments
- [Running the experiments](https://github.com/time-series-machine-learning/rist-pipeline-2023#running-the-experiments)
  with the latest versions or the versions used in the paper
- Estimator configurations for the
  [classifiers](https://github.com/time-series-machine-learning/rist-pipeline-2023/blob/main/set_rist_classifier.py)
  and
  [regressors](https://github.com/time-series-machine-learning/rist-pipeline-2023/blob/main/set_rist_regressor.py)
- [Example notebook](https://github.com/time-series-machine-learning/rist-pipeline-2023/blob/main/rist_pipeline.ipynb)
  using the RIST classifier and regressor

RIST is available in the [aeon](https://github.com/aeon-toolkit/aeon) Python package
as `RISTClassifier` and `RISTRegressor`.
