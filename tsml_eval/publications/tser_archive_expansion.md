# Unsupervised Feature Based Algorithms for Time Series Extrinsic Regression

David Guijo-Rubio, Matthew Middlehurst, Guilherme Arcencio, Diego Furtado Silva and
Anthony Bagnall. Data Mining and Knowledge Discovery, 2024.
[Paper](https://doi.org/10.1007/s10618-024-01027-w)

The paper expands the time series extrinsic regression (TSER) archive from 19 to 63
datasets and benchmarks regression algorithms on it, including the FreshPRINCE and
DrCIF regressors.

The code and results for the paper are in the
[tser-archive-expansion-2024](https://github.com/time-series-machine-learning/tser-archive-expansion-2024)
repository:

- [Results](https://github.com/time-series-machine-learning/tser-archive-expansion-2024/tree/main/results)
  for the 19 original datasets and the 63 datasets of the expanded archive
- [Datasets](https://github.com/time-series-machine-learning/tser-archive-expansion-2024#datasets)
  used in the experiments
- [Running the experiments](https://github.com/time-series-machine-learning/tser-archive-expansion-2024#running-the-experiments)
  with the latest versions or the versions used in the paper
- [Estimator configurations](https://github.com/time-series-machine-learning/tser-archive-expansion-2024/blob/main/set_tser_exp_regressor.py)
  for the regressors
- [Example notebook](https://github.com/time-series-machine-learning/tser-archive-expansion-2024/blob/main/tser_archive_expansion.ipynb)
  using the regressors

The regressors are available in the [aeon](https://github.com/aeon-toolkit/aeon) and
[tsml](https://github.com/time-series-machine-learning/tsml-py) Python packages.
