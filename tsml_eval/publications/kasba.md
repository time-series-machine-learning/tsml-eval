# Rock the KASBA: blazingly fast and accurate time series clustering

Christopher Holder and Anthony Bagnall. Data Mining and Knowledge Discovery, 2026.
[Paper](https://doi.org/10.1007/s10618-026-01189-9)

The paper introduces KASBA, a k-means clusterer that uses the move-split-merge
distance and a stochastic subgradient barycentre average, and compares it to other
time series clustering algorithms on 112 UCR datasets.

The code and results for the paper are in the
[kasba-clustering-2026](https://github.com/time-series-machine-learning/kasba-clustering-2026)
repository:

- [Results](https://github.com/time-series-machine-learning/kasba-clustering-2026/tree/main/results)
  for each experiment in Section 5 of the paper
- [Datasets](https://github.com/time-series-machine-learning/kasba-clustering-2026#datasets)
  used in the experiments
- [Running the experiments](https://github.com/time-series-machine-learning/kasba-clustering-2026#running-the-experiments)
  with the latest versions or the versions used in the paper
- [Clusterer configurations](https://github.com/time-series-machine-learning/kasba-clustering-2026/blob/main/_model_configuration.py)
  and a [table of parameters](https://github.com/time-series-machine-learning/kasba-clustering-2026#parameters)
- [Example notebook](https://github.com/time-series-machine-learning/kasba-clustering-2026/blob/main/kasba.ipynb)
  using KASBA, and a
  [notebook for the results figures](https://github.com/time-series-machine-learning/kasba-clustering-2026/blob/main/result_visualisation.ipynb)

KASBA and the other clusterers are available in the
[aeon](https://github.com/aeon-toolkit/aeon) Python package.
