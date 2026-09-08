"""Difficulty-controlled correlation heatmap for the interval review.

Per-dataset accuracies are dominated by how hard each dataset is, so raw
correlations between methods are all high and say little. Subtracting each
dataset's mean accuracy across the methods removes that shared component, and
what remains describes which methods do relatively well on the same problems.

The figure was previously a seaborn clustermap. This draws the same matrix in the
same clustered order but without the dendrograms, which carried no information
the ordering does not already convey.

Removing the dataset mean makes the deviations sum to zero across methods, which
constrains the correlations: with ``k`` methods the average off-diagonal
correlation is forced to ``-1 / (k - 1)``, about ``-0.11`` for ten. That null is
reported in the caption so the negative values are not read as evidence of
anticorrelation.

Usage::

    python correlation_heatmap.py --results <accuracy_mean.csv> --out <fig.pdf>
"""

__maintainer__ = ["TonyBagnall"]
__all__ = ["difficulty_controlled_correlations", "clustered_order", "plot"]

import argparse
import csv
import sys

import numpy as np

#: The order the methods appear in the collated file is alphabetical; the paper
#: uses the clustered order, which this module recomputes.
DEFAULT_RESULTS = (
    "D:/Results/UCR/Evaluation/Table1-30Resamples/Table1Univariate30/"
    "Accuracy/accuracy_mean.csv"
)
DEFAULT_OUT = (
    "C:/Users/Tony/OneDrive - University of Southampton/Research/Papers/WIP/"
    "Interval Based TSC/img/corr_clustermap.pdf"
)


def read_accuracies(path):
    """Return (method names, accuracy matrix of shape (n_datasets, n_methods))."""
    with open(path, encoding="utf-8") as fh:
        rows = list(csv.reader(fh))
    methods = rows[0][1:]
    values = []
    for row in rows[1:]:
        if not row or not row[0]:
            continue
        values.append([float(v) for v in row[1:]])
    return methods, np.array(values)


def difficulty_controlled_correlations(accuracies):
    """Correlate method profiles after removing each dataset's mean accuracy."""
    centred = accuracies - accuracies.mean(axis=1, keepdims=True)
    return np.corrcoef(centred, rowvar=False)


def null_correlation(n_methods):
    """The correlation induced by centring alone, with no real structure."""
    return -1.0 / (n_methods - 1)


def clustered_order(correlations):
    """Leaf order from average-linkage clustering, as the clustermap used."""
    from scipy.cluster.hierarchy import dendrogram, linkage
    from scipy.spatial.distance import pdist

    link = linkage(pdist(correlations, metric="euclidean"), method="average")
    return dendrogram(link, no_plot=True)["leaves"]


def plot(methods, correlations, order, out_path, null=None):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    names = [methods[i] for i in order]
    matrix = correlations[np.ix_(order, order)]
    n = len(names)

    fig, ax = plt.subplots(figsize=(7.6, 6.6))
    image = ax.imshow(matrix, cmap="RdBu_r", vmin=-1, vmax=1)

    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(names, rotation=90)
    ax.set_yticklabels(names)
    ax.tick_params(length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)

    # Gridlines between cells rather than a border around the block.
    ax.set_xticks(np.arange(-0.5, n, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, n, 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=1.5)
    ax.tick_params(which="minor", length=0)

    for i in range(n):
        for j in range(n):
            value = matrix[i, j]
            # White text on the saturated cells, dark on the pale ones.
            colour = "white" if abs(value) > 0.55 else "0.15"
            ax.text(j, i, "%.2f" % value, ha="center", va="center",
                    fontsize=8, color=colour)

    bar = fig.colorbar(image, ax=ax, shrink=0.72, ticks=[-1, 0, 1])
    bar.outline.set_visible(False)
    if null is not None:
        bar.ax.axhline(null, color="0.25", linewidth=1.0)
        bar.ax.text(1.6, null, " null %.2f" % null, va="center", fontsize=7,
                    transform=bar.ax.get_yaxis_transform())

    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    print("wrote", out_path)
    return fig


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", default=DEFAULT_RESULTS)
    parser.add_argument("--out", default=DEFAULT_OUT)
    parser.add_argument("--no-cluster", action="store_true",
                        help="keep the file's own method order")
    args = parser.parse_args(argv)

    methods, accuracies = read_accuracies(args.results)
    correlations = difficulty_controlled_correlations(accuracies)
    order = (list(range(len(methods))) if args.no_cluster
             else clustered_order(correlations))
    null = null_correlation(len(methods))

    print("%d datasets, %d methods" % (accuracies.shape[0], len(methods)))
    print("order: %s" % ", ".join(methods[i] for i in order))
    off = correlations[~np.eye(len(methods), dtype=bool)]
    print("mean off-diagonal correlation %.4f, induced null %.4f"
          % (off.mean(), null))

    plot(methods, correlations, order, args.out, null=null)
    return 0


if __name__ == "__main__":
    sys.exit(main())
