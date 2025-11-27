# 📘 KASBA: k-means Accelerated Stochastic Subgradient Barycentre Average
**Official Repository for the KASBA Time Series Clustering Paper**

This repository accompanies the paper:

> **Rock the KASBA: Blazingly Fast and Accurate Time Series Clustering**
>
> https://arxiv.org/abs/2411.17838

KASBA is a $k$-means clustering algorithm that uses the Move-Split-Merge (MSM) elastic distance at all stages of clustering, applies a randomised stochastic subgradient descent to find barycentre centroids, links each stage of clustering to accelerate convergence and exploits the metric property of MSM distance to avoid a large proportion of distance calculations. It is a versatile and scalable clusterer designed for real-world TSCL applications. It allows practitioners to balance  runtime and clustering performance when similarity is best measured by an elastic distance.

KASBA delivers state-of-the-art clustering performance while achieving 1–3 orders of magnitude speedups over existing elastic distance–based k-means algorithms.

This repository contains the exact model configurations, experiment scripts, and visualisation tools used to produce the results in the paper.

---

## 📁 Repository Structure

    kasba/
    ├── README.md                   # This file
    ├── __init__.py
    ├── _utils.py                   # Internal utilities used across the project
    ├── _model_configuration.py     # Definitions of all models and configurations used in experiments
    ├── _experiment_script.py       # Script used to run experiments on datasets
    ├── kasba.ipynb                 # Notebook demonstrating how to run KASBA
    ├── result_visualisation.ipynb  # Notebook for generating CD diagrams, MCM plots, etc.
    └── results/                    # Raw CSV result files used in the paper
        └── combined                # Subfolder for combined results
            └── k-shape-compare     # Subfolders results in section 5.4
            └── section-5.1         # Subfolders results in section 5.1
        └── train-test              # Subfolders for train and test results
            └── section-5.1         # Subfolders results in section 5.1
            └── section-5.2         # Subfolders results in section 5.2
            └── section-5.3         # Subfolders results in section 5.3


## 🚀 Getting Started

### Install dependencies

Create and activate a virtual environment from tsml-eval:

    python3 -m venv venv
    source venv/bin/activate
    pip install -e .

If you are reading this message you will have to install a specific branch
of aeon while we wait for a new release. Run the following command to install:

    pip uninstall aeon
    pip install git+https://github.com/aeon-toolkit/aeon@kasba-results#egg=aeon

Note: The project uses aeon, numpy, matplotlib, and other standard scientific Python packages.

---

## 🧪 Running KASBA

Minimal example from the kasba.ipynb notebook:

    from kasba import KASBA
    from aeon.datasets import load_dataset

    X, y = load_dataset("GunPoint")

    model = KASBA(
        n_clusters=2,
        distance="msm",
        distance_params={
            "c": 1.0
        },
    )

    labels = model.fit_predict(X)

The notebook demonstrates:

- How to use KASBA with different elastic distances
- How to cluster multivariate or unequal-length time series
- How to run multiple initialisations
- How to inspect convergence behaviour

---

## 📊 Reproducing Figures (CD & MCM)

Use the result_visualisation.ipynb notebook to generate:

- Critical Difference diagrams
- Model Comparison Matrices
- Ranking curves and statistical tests

---

## 📜 Citation

If you use KASBA in academic work, please cite the paper:

    C. Holder, A. Bagnall, Rock the kasba: Blazingly fast and accurate time
    series clustering, arXiv preprint arXiv:2411.17838 (2024)

(A full BibTeX entry will be added once the paper is published.)

---

## 🤝 Contact

For questions or queries please open an issue on tsml-eval.
