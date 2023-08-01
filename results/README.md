# tsml Results

This folder contains the results from various papers and projects from the tsml
research group.

We only store our most commonly used performance metric for each estimator/dataset
combination i.e. accuracy and root-mean-square error. Extended results may be available
on [timeseriesclassification.com](https://timeseriesclassification.com) or the
associated publication webpage the results are from.

We cannot guarantee that the results here are the same as those you will obtain by
running experiments using tsml-eval, as they have come from different sources, and the
method of resampling for datasets may differ (we aim to mitigate this over time by
updating results, but can never guarantee it).

Generally, all the results here use archive datasets from the UCR/UEA time series
classification repository or the Monash/UEA/UCR time series extrinsic regression
repository. We aim to provide results over 30 resamples, but this may not always be the
case and some classifiers may be missing results for certain datasets. Fold 0 in the
result files will be run using the provided train/test split for the dataset.

## Citations for Results

If you use any of these results in your work, please cite the relevant paper. We list
some common results and data sources below, but they may not be relevant to each
individual estimator result file.

Univariate classification bakeoff 2023:
- Middlehurst, M., Schäfer, P. and Bagnall, A., 2023. Bake off redux: a review and
  experimental evaluation of recent time series classification algorithms. arXiv
  preprint arXiv:2304.13029.

Multivariate classification bakeoff 2021:
- Ruiz, A.P., Flynn, M., Large, J., Middlehurst, M. and Bagnall, A., 2021. The great
  multivariate time series classification bake off: a review and experimental evaluation
  of recent algorithmic advances. Data Mining and Knowledge Discovery, 35(2),
  pp.401-449.

Distance-based classification 2023:
- Holder, C., Middlehurst, M. and Bagnall, A., 2022. A review and evaluation of elastic
  distance functions for time series clustering. arXiv preprint arXiv:2205.15181.

Extrinsic regression 63 datasets results 2023:
- Guijo-Rubio, D., Middlehurst, M., Arcencio, G., Silva, D.F. and Bagnall, A., 2023.
  Unsupervised feature based algorithms for time series extrinsic regression.
  arXiv preprint arXiv:2305.01429.

128 UCR univariate datasets:
- Dau, H.A., Bagnall, A., Kamgar, K., Yeh, C.C.M., Zhu, Y., Gharghabi, S.,
  Ratanamahatana, C.A. and Keogh, E., 2019. The UCR time series archive. IEEE/CAA
  Journal of Automatica Sinica, 6(6), pp.1293-1305.

30 UEA multivariate datasets:
- Bagnall, A., Dau, H.A., Lines, J., Flynn, M., Large, J., Bostrom, A., Southam, P. and
  Keogh, E., 2018. The UEA multivariate time series classification archive, 2018.
  arXiv preprint arXiv:1811.00075.

19 Monash/UEA/UCR  extrinsic regression datasets:
- Tan, C.W., Bergmeir, C., Petitjean, F. and Webb, G.I., 2020. Monash University, UEA,
  UCR time series extrinsic regression archive. arXiv preprint arXiv:2006.10996.

Extended extrinsic regression archive 63 datasets:
- Guijo-Rubio, D., Middlehurst, M., Arcencio, G., Silva, D.F. and Bagnall, A., 2023.
  Unsupervised feature based algorithms for time series extrinsic regression.
  arXiv preprint arXiv:2305.01429.
