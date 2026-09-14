"""Cross-validation utilities for dependent time-series cases."""

__maintainer__ = ["MatthewMiddlehurst"]
__all__ = [
    "BufferedBlockedKFold",
    "HVBlockCrossValidator",
]

from numbers import Integral

import numpy as np
from sklearn.model_selection import BaseCrossValidator
from sklearn.utils import indexable
from sklearn.utils.validation import _num_samples


def _validate_non_negative_integer(value, name):
    """Validate and return a non-negative integer parameter."""
    if (
        isinstance(value, (bool, np.bool_))
        or not isinstance(value, Integral)
        or value < 0
    ):
        raise ValueError(f"{name} must be a non-negative integer, but found {value!r}.")
    return int(value)


def _validate_n_splits(n_splits):
    """Validate and return the number of folds."""
    if (
        isinstance(n_splits, (bool, np.bool_))
        or not isinstance(n_splits, Integral)
        or n_splits < 2
    ):
        raise ValueError(
            "n_splits must be an integer greater than or equal to 2, "
            f"but found {n_splits!r}."
        )
    return int(n_splits)


def _get_sequence_indices(n_samples, groups):
    """Return input-order indices for each independent sequence."""
    if groups is None:
        return [np.arange(n_samples, dtype=int)]

    groups = np.asarray(groups)
    if groups.ndim != 1:
        raise ValueError("groups must be one-dimensional.")

    sequence_ids, inverse = np.unique(groups, return_inverse=True)
    return [
        np.flatnonzero(inverse == sequence_index)
        for sequence_index in range(len(sequence_ids))
    ]


def _get_exclusion_bounds(test_start, test_stop, sequence_size, h):
    """Return the test block and buffer bounds within a sequence."""
    return max(0, test_start - h), min(sequence_size, test_stop + h)


def _yield_sliding_block_splits(n_samples, sequence_indices, h, v):
    """Yield two-sided buffered splits for every valid sliding test block."""
    test_size = 2 * v + 1
    split_specs = []
    parameters = "h" if v == 0 else "h or v"
    for indices in sequence_indices:
        for test_start in range(len(indices) - test_size + 1):
            test_stop = test_start + test_size
            excluded_start, excluded_stop = _get_exclusion_bounds(
                test_start, test_stop, len(indices), h
            )
            if excluded_stop - excluded_start == n_samples:
                raise ValueError(
                    f"A split has no training cases. Reduce {parameters}, or provide "
                    "more cases or independent sequences."
                )
            split_specs.append(
                (
                    indices[test_start],
                    indices,
                    test_start,
                    test_stop,
                    excluded_start,
                    excluded_stop,
                )
            )

    split_specs.sort(key=lambda spec: spec[0])
    for _, indices, test_start, test_stop, excluded_start, excluded_stop in split_specs:
        train_mask = np.ones(n_samples, dtype=bool)
        train_mask[indices[excluded_start:excluded_stop]] = False
        train = np.flatnonzero(train_mask)
        yield train, indices[test_start:test_stop]


def _validate_test_size(sequence_indices, test_size):
    """Validate that every sequence can contain a complete test block."""
    min_sequence_size = min(len(indices) for indices in sequence_indices)
    if min_sequence_size < test_size:
        raise ValueError(
            "Each sequence must contain at least 2 * v + 1 cases. "
            f"Found a test block size of {test_size} and a smallest sequence "
            f"size of {min_sequence_size}."
        )


class HVBlockCrossValidator(BaseCrossValidator):
    """Sliding hv-block cross-validator for dependent observations.

    Every contiguous block of ``2v + 1`` cases is used as the test set once. For
    each test block, the ``h`` cases immediately before it and the ``h`` cases
    immediately after it are excluded from training. Consecutive test blocks overlap
    when ``v > 0``. A sequence of length ``n`` therefore produces ``n - 2v`` folds.

    When ``v=0``, each test block contains one case and the splits are the h-block
    procedure proposed by Burman et al. [1]_. Scores calculated directly from these
    splits are uncorrected. Burman et al. also proposed a finite-sample correction for
    the estimated prediction error, which belongs in score calculation rather than in
    this splitter.

    For ``v>0``, the split geometry for one ordered sequence is the hv-block procedure
    described by Racine [2]_. ``groups`` extends both procedures to multiple
    independent sequences by applying the sliding procedure separately within each
    sequence. Each fold tests one block from one sequence; cases from other sequences
    remain available for training.

    Zheng [3]_ showed that the balanced incomplete block design used in Racine's
    consistency argument does not hold, leaving that theoretical guarantee open.

    Parameters
    ----------
    h : int, default=0
        Number of neighbouring cases to exclude from training on each side of the
        test block.
    v : int, default=0
        Number of cases on each side of the centre case that form the test block.
        The complete test block therefore contains ``2v + 1`` cases. ``v=0`` gives
        h-block cross-validation.

    References
    ----------
    .. [1] Burman, P., Chow, E., and Nolan, D. (1994). A cross-validatory
       method for dependent data. Biometrika, 81(2), 351-358.
       https://doi.org/10.1093/biomet/81.2.351
    .. [2] Racine, J. (2000). Consistent cross-validatory model-selection for
       dependent data: hv-block cross-validation. Journal of Econometrics, 99(1),
       39-61. https://doi.org/10.1016/S0304-4076(00)00030-0
    .. [3] Zheng, W. (2019). hv-Block Cross Validation is not a BIBD: a Note on
       the Paper by Jeff Racine (2000). https://arxiv.org/abs/1910.08904

    Examples
    --------
    >>> import numpy as np
    >>> from tsml_eval.utils.cross_validation import HVBlockCrossValidator
    >>> X = np.arange(8)
    >>> cv = HVBlockCrossValidator(h=1, v=1)
    >>> for train, test in cv.split(X):
    ...     print(train, test)
    [4 5 6 7] [0 1 2]
    [5 6 7] [1 2 3]
    [0 6 7] [2 3 4]
    [0 1 7] [3 4 5]
    [0 1 2] [4 5 6]
    [0 1 2 3] [5 6 7]
    """

    # Declare that groups are consumed when sklearn metadata routing is enabled.
    __metadata_request__split = {"groups": True}

    def __init__(self, h=0, v=0):
        self.h = _validate_non_negative_integer(h, "h")
        self.v = _validate_non_negative_integer(v, "v")

    def get_n_splits(self, X, y=None, groups=None):
        """Return the number of valid sliding test blocks.

        Parameters
        ----------
        X : array-like of shape (n_cases, ...)
            Data used to determine the number of cases.
        y : array-like of shape (n_cases,), default=None
            Target values. Not used.
        groups : array-like of shape (n_cases,), default=None
            Independent sequence ID for each case. If supplied, the total is the sum
            of the valid test blocks across all sequences.

        Returns
        -------
        n_splits : int
            Total number of valid sliding test blocks.
        """
        X, y, groups = indexable(X, y, groups)
        n_samples = _num_samples(X)
        if n_samples == 0:
            raise ValueError("HVBlockCrossValidator requires at least one case.")

        sequence_indices = _get_sequence_indices(n_samples, groups)
        test_size = 2 * self.v + 1
        _validate_test_size(sequence_indices, test_size)
        return sum(len(indices) - 2 * self.v for indices in sequence_indices)

    def split(self, X, y=None, groups=None):
        """Generate sliding hv-block train and test indices.

        Parameters
        ----------
        X : array-like of shape (n_cases, ...)
            Data used to determine the number of cases.
        y : array-like of shape (n_cases,), default=None
            Target values. Not used.
        groups : array-like of shape (n_cases,), default=None
            Independent sequence ID for each case. Cases with the same ID are treated
            as one ordered sequence, using their order in the input. Sliding test
            blocks and their exclusion buffers stay within that sequence. If
            ``None``, all cases are treated as one continuous sequence. The cases are
            not sorted or shuffled, so they must already be in chronological order
            within each sequence.

        Yields
        ------
        train : np.ndarray
            Training case indices.
        test : np.ndarray
            Test block indices.
        """
        X, y, groups = indexable(X, y, groups)
        n_samples = _num_samples(X)
        if n_samples < 2:
            raise ValueError(
                "HVBlockCrossValidator requires at least 2 cases, "
                f"but found {n_samples}."
            )

        sequence_indices = _get_sequence_indices(n_samples, groups)
        _validate_test_size(sequence_indices, 2 * self.v + 1)
        yield from _yield_sliding_block_splits(
            n_samples, sequence_indices, h=self.h, v=self.v
        )


class BufferedBlockedKFold(BaseCrossValidator):
    """Non-overlapping blocked cross-validator with two-sided buffering.

    The input cases may come from one sequence or from several independent sequences.
    Every sequence is divided into ``n_splits`` contiguous test blocks. Fold ``k``
    combines block ``k`` from every sequence, while the ``gap`` neighbouring cases
    on either side of each test block are excluded from training. Within a sequence,
    partitions differ in size by at most one case.

    Pass one sequence ID per case through ``groups`` in :meth:`split` when cases come
    from multiple sequences. This prevents the end of one sequence from being treated
    as adjacent to the start of another.

    Unlike :class:`HVBlockCrossValidator`, the test blocks do not overlap and every
    case is tested exactly once.

    Parameters
    ----------
    n_splits : int, default=5
        Number of non-overlapping test partitions. Every sequence must contain at
        least ``n_splits`` cases so that it contributes to every fold.
    gap : int, default=0
        Number of neighbouring cases to exclude from training on each side of every
        test block.

    Examples
    --------
    >>> import numpy as np
    >>> from tsml_eval.utils.cross_validation import BufferedBlockedKFold
    >>> X = np.arange(10)
    >>> groups = np.repeat(["sequence_a", "sequence_b"], 5)
    >>> cv = BufferedBlockedKFold(n_splits=2, gap=1)
    >>> for train, test in cv.split(X, groups=groups):
    ...     print(train, test)
    [4 9] [0 1 2 5 6 7]
    [0 1 5 6] [3 4 8 9]
    """

    # Declare that groups are consumed when sklearn metadata routing is enabled.
    __metadata_request__split = {"groups": True}

    def __init__(self, n_splits=5, gap=0):
        self.n_splits = _validate_n_splits(n_splits)
        self.gap = _validate_non_negative_integer(gap, "gap")

    def get_n_splits(self, X=None, y=None, groups=None):
        """Return the configured number of folds.

        Parameters
        ----------
        X : array-like of shape (n_cases, ...), default=None
            Not used.
        y : array-like of shape (n_cases,), default=None
            Not used.
        groups : array-like of shape (n_cases,), default=None
            Not used.

        Returns
        -------
        n_splits : int
            Number of folds given by ``n_splits`` at construction.
        """
        return self.n_splits

    def split(self, X, y=None, groups=None):
        """Generate buffered blocked train and test indices.

        Parameters
        ----------
        X : array-like of shape (n_cases, ...)
            Data used to determine the number of cases.
        y : array-like of shape (n_cases,), default=None
            Target values. Not used.
        groups : array-like of shape (n_cases,), default=None
            Independent sequence ID for each case. Cases with the same ID are treated
            as one ordered sequence, using their order in the input. Each sequence is
            partitioned separately, and fold ``k`` tests partition ``k`` from every
            sequence. Exclusion buffers stay within their sequence, while distant
            cases from the same sequence may remain in training. Each ID must occur
            at least ``n_splits`` times. If ``None``, all cases are treated as one
            continuous sequence. The cases are not sorted or shuffled, so they must
            already be in chronological order within each sequence.

        Yields
        ------
        train : np.ndarray
            Training case indices.
        test : np.ndarray
            Test case indices combined across sequences.
        """
        X, y, groups = indexable(X, y, groups)
        n_samples = _num_samples(X)
        if n_samples == 0:
            raise ValueError("BufferedBlockedKFold requires at least one case.")

        sequence_indices = _get_sequence_indices(n_samples, groups)

        min_sequence_size = min(len(indices) for indices in sequence_indices)
        if min_sequence_size < self.n_splits:
            raise ValueError(
                "Each sequence must contain at least n_splits cases so that it "
                "contributes to every test fold. "
                f"Found n_splits={self.n_splits} and a smallest sequence size of "
                f"{min_sequence_size}."
            )

        sequence_blocks = [
            np.array_split(np.arange(len(indices), dtype=int), self.n_splits)
            for indices in sequence_indices
        ]

        fold_specs = []
        for fold in range(self.n_splits):
            excluded_cases = 0
            sequence_specs = []
            for indices, blocks in zip(sequence_indices, sequence_blocks):
                block_positions = blocks[fold]
                excluded_start, excluded_stop = _get_exclusion_bounds(
                    block_positions[0],
                    block_positions[-1] + 1,
                    len(indices),
                    self.gap,
                )
                excluded_cases += excluded_stop - excluded_start
                sequence_specs.append(
                    (indices, block_positions, excluded_start, excluded_stop)
                )

            if excluded_cases == n_samples:
                raise ValueError(
                    f"Fold {fold} has no training cases. Reduce gap, increase "
                    "n_splits, or provide more cases per sequence."
                )
            fold_specs.append(sequence_specs)

        for sequence_specs in fold_specs:
            train_mask = np.ones(n_samples, dtype=bool)
            test_parts = []

            for (
                indices,
                block_positions,
                excluded_start,
                excluded_stop,
            ) in sequence_specs:
                test_parts.append(indices[block_positions])
                train_mask[indices[excluded_start:excluded_stop]] = False

            train = np.flatnonzero(train_mask)
            test = np.sort(np.concatenate(test_parts))
            yield train, test
