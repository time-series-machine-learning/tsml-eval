# Extracting Features from Random Subseries: A Hybrid Pipeline for Time Series Classification and Extrinsic Regression

Matthew Middlehurst and Anthony Bagnall. International Workshop on Advanced Analytics
and Learning on Temporal Data (AALTD), 2023.
[Paper](https://doi.org/10.1007/978-3-031-49896-1_8)

The code, results files and instructions for running the experiments are available at
<https://github.com/time-series-machine-learning/rist-pipeline-2023>.

## Summary

Approaches which combine features from more than one domain, such as HIVE-COTE and
TS-CHIEF, generally perform better than single domain approaches for time series
classification. The paper asks whether the same is true for time series extrinsic
regression, which has seen far less research.

It introduces the Randomised Interval-Shapelet Transformation (RIST) pipeline. RIST
extracts features from random intervals and random dilated shapelets taken from
several representations of the series, concatenates them, and builds an extra trees
estimator on the result.

On the 112 UCR classification datasets and 63 regression datasets, RIST significantly
outperforms the single domain approaches. The paper also provides results for shapelet
based regression using the RDST transform, and shows that interval based approaches
such as DrCIF scale noticeably better in a pipeline format.

RIST is available in the [aeon](https://github.com/aeon-toolkit/aeon) Python package
as `RISTClassifier` and `RISTRegressor`.
