# Bake off redux: a review and experimental evaluation of recent time series classification algorithms

Matthew Middlehurst, Patrick Schäfer and Anthony Bagnall. Data Mining and Knowledge
Discovery, 2024.
[Paper](https://doi.org/10.1007/s10618-024-01022-1),
[Correction](https://doi.org/10.1007/s10618-024-01040-z)

The paper reviews recent time series classification algorithms and compares 40 of them
on 112 UCR datasets and 30 new datasets.

The code and results for the paper are in the
[tsc-bakeoff-2024](https://github.com/time-series-machine-learning/tsc-bakeoff-2024)
repository:

- [Results](https://github.com/time-series-machine-learning/tsc-bakeoff-2024/tree/main/results)
  for each classifier category on the 112 UCR datasets, and for the best classifiers on
  all 142 datasets
- [Datasets](https://github.com/time-series-machine-learning/tsc-bakeoff-2024#datasets)
  used in the experiments
- [Running the experiments](https://github.com/time-series-machine-learning/tsc-bakeoff-2024#running-the-experiments)
  with the latest versions or the versions used in the paper, including the Java
  classifiers and resamples
- [Estimator configurations](https://github.com/time-series-machine-learning/tsc-bakeoff-2024/blob/main/set_bakeoff_classifier.py)
  and a [table of classifier parameters](https://github.com/time-series-machine-learning/tsc-bakeoff-2024#classifier-parameters)
- [Example notebook](https://github.com/time-series-machine-learning/tsc-bakeoff-2024/blob/main/tsc_bakeoff_2023.ipynb)
  using the classifiers

Most of the classifiers are available in the [aeon](https://github.com/aeon-toolkit/aeon)
and [tsml](https://github.com/time-series-machine-learning/tsml-py) Python packages.
