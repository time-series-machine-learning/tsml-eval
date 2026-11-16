# A review and evaluation of elastic distance functions for time series clustering

Christopher Holder, Matthew Middlehurst and Anthony Bagnall. Knowledge and Information
Systems, 2024.
[Paper](https://doi.org/10.1007/s10115-023-01952-0)

The paper reviews nine elastic distance functions and compares them, with Euclidean
distance, using k-means and k-medoids clustering on 112 UCR datasets.

The code and results for the paper are in the
[elastic-distance-clustering-2023](https://github.com/time-series-machine-learning/elastic-distance-clustering-2023)
repository:

- [Results](https://github.com/time-series-machine-learning/elastic-distance-clustering-2023/tree/main/results)
  for each experiment in Section 5 of the paper
- [Datasets](https://github.com/time-series-machine-learning/elastic-distance-clustering-2023#datasets)
  used in the experiments
- [Running the experiments](https://github.com/time-series-machine-learning/elastic-distance-clustering-2023#running-the-experiments)
  with the latest versions or the versions used in the paper
- [Clusterer configurations](https://github.com/time-series-machine-learning/elastic-distance-clustering-2023/blob/main/set_distance_clusterer.py)
  and a [table of distance parameters](https://github.com/time-series-machine-learning/elastic-distance-clustering-2023#parameters)
- [Example notebook](https://github.com/time-series-machine-learning/elastic-distance-clustering-2023/blob/main/distance_based_clustering.ipynb)
  using the clusterers, and notebooks for the
  [alignment path figures](https://github.com/time-series-machine-learning/elastic-distance-clustering-2023/blob/main/alignment_and_paths_figures.ipynb)
  and the [distance timings](https://github.com/time-series-machine-learning/elastic-distance-clustering-2023/blob/main/package_distance_timing.ipynb)

The clusterers and distances are available in the
[aeon](https://github.com/aeon-toolkit/aeon) Python package.
