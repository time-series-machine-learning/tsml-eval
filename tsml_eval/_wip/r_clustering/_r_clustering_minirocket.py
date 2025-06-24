"""MiniRocket transformer for r-clustering.

Code adapted from:
https://github.com/jorgemarcoes/R-Clustering/blob/main/R_Clustering_on_UCR_Archive.ipynb
"""

__all__ = ["RClusteringTransformer"]

import multiprocessing
from typing import Union

import numpy as np
from aeon.transformations.collection import BaseCollectionTransformer
from numba import get_num_threads, njit, prange, set_num_threads, vectorize
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


class RClusteringTransformer(BaseCollectionTransformer):

    _tags = {
        "output_data_type": "Tabular",
        "algorithm_type": "convolution",
    }

    def __init__(
        self,
        num_features: int = 500,
        max_dilations_per_kernel: int = 32,
        pca_result: bool = True,
        n_jobs=1,
        random_state=None,
    ):
        self.num_features = num_features
        self.max_dilations_per_kernel = max_dilations_per_kernel
        self.pca_result = pca_result

        self.n_jobs = n_jobs
        self.random_state = random_state
        self._dim_reduction_transformer = None
        super().__init__()

    def _fit(self, X, y=None):
        random_state = (
            np.int32(self.random_state) if isinstance(self.random_state, int) else None
        )
        X = X.squeeze()
        X = X.astype(np.float32)
        _, n_timepoints = X.shape
        if n_timepoints < 9:
            raise ValueError(
                f"n_timepoints must be >= 9, but found {n_timepoints};"
                " zero pad shorter series so that n_timepoints == 9"
            )
        self.parameters = _fit(
            X, self.num_features, self.max_dilations_per_kernel, random_state
        )
        return self

    def _transform(self, X, y=None):
        X = X.squeeze()
        X = X.astype(np.float32)

        # change n_jobs dependend on value and existing cores
        prev_threads = get_num_threads()
        if self.n_jobs < 1 or self.n_jobs > multiprocessing.cpu_count():
            n_jobs = multiprocessing.cpu_count()
        else:
            n_jobs = self.n_jobs
        set_num_threads(n_jobs)

        X_ = _transform(X, self.parameters)

        if self.pca_result:
            sc = StandardScaler()
            X_std = sc.fit_transform(X_)

            if self._dim_reduction_transformer is None:
                pca = PCA(random_state=self.random_state).fit(X_std)
                optimal_dimensions = np.argmax(pca.explained_variance_ratio_ < 0.01)
                pca_optimal = PCA(n_components=optimal_dimensions)
                self._dim_reduction_transformer = pca_optimal
            X_t = self._dim_reduction_transformer.fit_transform(X_std)
        else:
            X_t = X_

        set_num_threads(prev_threads)

        return X_t


@njit(
    "float32[:](float32[:,:],int32[:],int32[:],float32[:])",
    fastmath=True,
    parallel=False,
    cache=True,
)
def _fit_biases(X, dilations, num_features_per_dilation, quantiles):
    num_examples, input_length = X.shape

    # R-clustering specific modification
    indices = np.array(
        (
            1,
            3,
            6,
            1,
            2,
            7,
            1,
            2,
            3,
            0,
            2,
            3,
            1,
            4,
            5,
            0,
            1,
            3,
            3,
            5,
            6,
            0,
            1,
            2,
            2,
            5,
            8,
            1,
            3,
            7,
            0,
            1,
            8,
            4,
            6,
            7,
            0,
            1,
            4,
            3,
            4,
            6,
            0,
            4,
            5,
            2,
            6,
            7,
            5,
            6,
            7,
            0,
            1,
            6,
            4,
            5,
            7,
            4,
            7,
            8,
            1,
            6,
            8,
            0,
            2,
            6,
            5,
            6,
            8,
            2,
            5,
            7,
            0,
            1,
            7,
            0,
            7,
            8,
            0,
            3,
            5,
            0,
            3,
            7,
            2,
            3,
            8,
            2,
            3,
            4,
            1,
            4,
            6,
            3,
            4,
            5,
            0,
            3,
            8,
            4,
            5,
            8,
            0,
            4,
            6,
            1,
            4,
            8,
            6,
            7,
            8,
            4,
            6,
            8,
            0,
            3,
            4,
            1,
            3,
            4,
            1,
            5,
            7,
            1,
            4,
            7,
            1,
            2,
            8,
            0,
            6,
            7,
            1,
            6,
            7,
            1,
            3,
            5,
            0,
            1,
            5,
            0,
            4,
            8,
            4,
            5,
            6,
            0,
            2,
            5,
            3,
            5,
            7,
            0,
            2,
            4,
            2,
            6,
            8,
            2,
            3,
            7,
            2,
            5,
            6,
            2,
            4,
            8,
            0,
            2,
            7,
            3,
            6,
            8,
            2,
            3,
            6,
            3,
            7,
            8,
            0,
            5,
            8,
            1,
            2,
            6,
            2,
            3,
            5,
            1,
            5,
            8,
            3,
            6,
            7,
            3,
            4,
            7,
            0,
            4,
            7,
            3,
            5,
            8,
            2,
            4,
            5,
            1,
            2,
            5,
            2,
            7,
            8,
            2,
            4,
            6,
            0,
            5,
            6,
            3,
            4,
            8,
            0,
            6,
            8,
            2,
            4,
            7,
            0,
            2,
            8,
            0,
            3,
            6,
            5,
            7,
            8,
            1,
            5,
            6,
            1,
            2,
            4,
            0,
            5,
            7,
            1,
            3,
            8,
            1,
            7,
            8,
        ),
        dtype=np.int32,
    ).reshape(84, 3)

    num_kernels = len(indices)
    num_dilations = len(dilations)

    num_features = num_kernels * np.sum(num_features_per_dilation)

    biases = np.zeros(num_features, dtype=np.float32)

    feature_index_start = 0

    for dilation_index in range(num_dilations):

        dilation = dilations[dilation_index]
        padding = ((9 - 1) * dilation) // 2

        num_features_this_dilation = num_features_per_dilation[dilation_index]

        for kernel_index in range(num_kernels):

            feature_index_end = feature_index_start + num_features_this_dilation

            _X = X[np.random.randint(num_examples)]

            A = -_X  # A = alpha * X = -X
            G = _X + _X + _X  # G = gamma * X = 3X

            C_alpha = np.zeros(input_length, dtype=np.float32)
            C_alpha[:] = A

            C_gamma = np.zeros((9, input_length), dtype=np.float32)
            C_gamma[9 // 2] = G

            start = dilation
            end = input_length - padding

            for gamma_index in range(9 // 2):
                C_alpha[-end:] = C_alpha[-end:] + A[:end]
                C_gamma[gamma_index, -end:] = G[:end]

                end += dilation

            for gamma_index in range(9 // 2 + 1, 9):
                C_alpha[:-start] = C_alpha[:-start] + A[start:]
                C_gamma[gamma_index, :-start] = G[start:]

                start += dilation

            index_0, index_1, index_2 = indices[kernel_index]

            C = C_alpha + C_gamma[index_0] + C_gamma[index_1] + C_gamma[index_2]

            biases[feature_index_start:feature_index_end] = np.quantile(
                C, quantiles[feature_index_start:feature_index_end]
            )

            feature_index_start = feature_index_end

    return biases


def _fit_dilations(input_length, num_features, max_dilations_per_kernel):
    num_kernels = 84

    num_features_per_kernel = num_features // num_kernels
    true_max_dilations_per_kernel = min(
        num_features_per_kernel, max_dilations_per_kernel
    )
    multiplier = num_features_per_kernel / true_max_dilations_per_kernel

    max_exponent = np.log2((input_length - 1) / (9 - 1))
    dilations, num_features_per_dilation = np.unique(
        np.logspace(0, max_exponent, true_max_dilations_per_kernel, base=2).astype(
            np.int32
        ),
        return_counts=True,
    )
    num_features_per_dilation = (num_features_per_dilation * multiplier).astype(
        np.int32
    )  # this is a vector

    remainder = num_features_per_kernel - np.sum(num_features_per_dilation)
    i = 0
    while remainder > 0:
        num_features_per_dilation[i] += 1
        remainder -= 1
        i = (i + 1) % len(num_features_per_dilation)

    return dilations, num_features_per_dilation


# low-discrepancy sequence to assign quantiles to kernel/dilation combinations
def _quantiles(n):
    return np.array(
        [(_ * ((np.sqrt(5) + 1) / 2)) % 1 for _ in range(1, n + 1)], dtype=np.float32
    )


def _fit(
    X, num_features=10_000, max_dilations_per_kernel=32, seed: Union[int, None] = None
):
    if seed is not None:
        np.random.seed(seed)

    _, input_length = X.shape

    num_kernels = 84

    dilations, num_features_per_dilation = _fit_dilations(
        input_length, num_features, max_dilations_per_kernel
    )

    num_features_per_kernel = np.sum(num_features_per_dilation)

    quantiles = _quantiles(num_kernels * num_features_per_kernel)

    # R-clustering specific modifications
    quantiles = np.random.permutation(quantiles)

    biases = _fit_biases(X, dilations, num_features_per_dilation, quantiles)

    return dilations, num_features_per_dilation, biases


# _PPV(C, b).mean() returns PPV for vector C (convolution output) and scalar b (bias)
@vectorize("float32(float32,float32)", nopython=True, cache=True)
def _PPV(a, b):
    if a > b:
        return 1
    else:
        return 0


@njit(
    "float32[:,:](float32[:,:],Tuple((int32[:],int32[:],float32[:])))",
    fastmath=True,
    parallel=True,
    cache=True,
)
def _transform(X, parameters):
    num_examples, input_length = X.shape

    dilations, num_features_per_dilation, biases = parameters

    # R-clustering specific modifications
    indices = np.array(
        (
            1,
            3,
            6,
            1,
            2,
            7,
            1,
            2,
            3,
            0,
            2,
            3,
            1,
            4,
            5,
            0,
            1,
            3,
            3,
            5,
            6,
            0,
            1,
            2,
            2,
            5,
            8,
            1,
            3,
            7,
            0,
            1,
            8,
            4,
            6,
            7,
            0,
            1,
            4,
            3,
            4,
            6,
            0,
            4,
            5,
            2,
            6,
            7,
            5,
            6,
            7,
            0,
            1,
            6,
            4,
            5,
            7,
            4,
            7,
            8,
            1,
            6,
            8,
            0,
            2,
            6,
            5,
            6,
            8,
            2,
            5,
            7,
            0,
            1,
            7,
            0,
            7,
            8,
            0,
            3,
            5,
            0,
            3,
            7,
            2,
            3,
            8,
            2,
            3,
            4,
            1,
            4,
            6,
            3,
            4,
            5,
            0,
            3,
            8,
            4,
            5,
            8,
            0,
            4,
            6,
            1,
            4,
            8,
            6,
            7,
            8,
            4,
            6,
            8,
            0,
            3,
            4,
            1,
            3,
            4,
            1,
            5,
            7,
            1,
            4,
            7,
            1,
            2,
            8,
            0,
            6,
            7,
            1,
            6,
            7,
            1,
            3,
            5,
            0,
            1,
            5,
            0,
            4,
            8,
            4,
            5,
            6,
            0,
            2,
            5,
            3,
            5,
            7,
            0,
            2,
            4,
            2,
            6,
            8,
            2,
            3,
            7,
            2,
            5,
            6,
            2,
            4,
            8,
            0,
            2,
            7,
            3,
            6,
            8,
            2,
            3,
            6,
            3,
            7,
            8,
            0,
            5,
            8,
            1,
            2,
            6,
            2,
            3,
            5,
            1,
            5,
            8,
            3,
            6,
            7,
            3,
            4,
            7,
            0,
            4,
            7,
            3,
            5,
            8,
            2,
            4,
            5,
            1,
            2,
            5,
            2,
            7,
            8,
            2,
            4,
            6,
            0,
            5,
            6,
            3,
            4,
            8,
            0,
            6,
            8,
            2,
            4,
            7,
            0,
            2,
            8,
            0,
            3,
            6,
            5,
            7,
            8,
            1,
            5,
            6,
            1,
            2,
            4,
            0,
            5,
            7,
            1,
            3,
            8,
            1,
            7,
            8,
        ),
        dtype=np.int32,
    ).reshape(84, 3)

    num_kernels = len(indices)
    num_dilations = len(dilations)

    num_features = num_kernels * np.sum(num_features_per_dilation)

    features = np.zeros((num_examples, num_features), dtype=np.float32)

    for example_index in prange(num_examples):

        _X = X[example_index]

        A = -_X  # A = alpha * X = -X
        G = _X + _X + _X  # G = gamma * X = 3X

        feature_index_start = 0

        for dilation_index in range(num_dilations):

            _padding0 = dilation_index % 2

            dilation = dilations[dilation_index]
            padding = ((9 - 1) * dilation) // 2

            num_features_this_dilation = num_features_per_dilation[dilation_index]

            C_alpha = np.zeros(input_length, dtype=np.float32)
            C_alpha[:] = A

            C_gamma = np.zeros((9, input_length), dtype=np.float32)
            C_gamma[9 // 2] = G

            start = dilation
            end = input_length - padding

            for gamma_index in range(9 // 2):
                C_alpha[-end:] = C_alpha[-end:] + A[:end]
                C_gamma[gamma_index, -end:] = G[:end]

                end += dilation

            for gamma_index in range(9 // 2 + 1, 9):
                C_alpha[:-start] = C_alpha[:-start] + A[start:]
                C_gamma[gamma_index, :-start] = G[start:]

                start += dilation

            for kernel_index in range(num_kernels):

                feature_index_end = feature_index_start + num_features_this_dilation

                _padding1 = (_padding0 + kernel_index) % 2

                index_0, index_1, index_2 = indices[kernel_index]

                C = C_alpha + C_gamma[index_0] + C_gamma[index_1] + C_gamma[index_2]

                if _padding1 == 0:
                    for feature_count in range(num_features_this_dilation):
                        features[example_index, feature_index_start + feature_count] = (
                            _PPV(C, biases[feature_index_start + feature_count]).mean()
                        )
                else:
                    for feature_count in range(num_features_this_dilation):
                        features[example_index, feature_index_start + feature_count] = (
                            _PPV(
                                C[padding:-padding],
                                biases[feature_index_start + feature_count],
                            ).mean()
                        )

                feature_index_start = feature_index_end

    return features
