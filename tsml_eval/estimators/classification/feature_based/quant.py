"""QUANT: A Minimalist Interval Method for Time Series Classification.

Angus Dempster, Daniel F Schmidt, Geoffrey I Webb
https://arxiv.org/abs/2308.00928

Original code: https://github.com/angus924/quant
"""

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.base import ClassifierMixin, TransformerMixin
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.utils.multiclass import check_classification_targets
from sklearn.utils.validation import check_is_fitted
from tsml.base import BaseTimeSeriesEstimator, _clone_estimator


class QuantClassifier(ClassifierMixin, BaseTimeSeriesEstimator):
    """
    QUANT classifier.

    Examples
    --------
    >>> from tsml_eval.estimators.classification.feature_based.quant import (
    ...     QuantClassifier)
    >>> from tsml.datasets import load_minimal_chinatown
    >>> X, y = load_minimal_chinatown()
    >>> classifier = QuantClassifier()
    >>> classifier = classifier.fit(X, y)
    >>> y_pred = classifier.predict(X)
    """

    def __init__(self, depth=6, div=4, estimator=None, random_state=None):
        self.depth = depth
        self.div = div
        self.estimator = estimator
        self.random_state = random_state

        super(QuantClassifier, self).__init__()

    def fit(self, X, y):
        """Fit the estimator to training data.

        Parameters
        ----------
        X : 3D np.ndarray of shape (n_instances, n_channels, n_timepoints)
            The training data.
        y : 1D np.ndarray of shape (n_instances)
            The class labels for fitting, indices correspond to instance indices in X

        Returns
        -------
        self :
            Reference to self.
        """
        X, y = self._validate_data(X=X, y=y, ensure_min_samples=2)
        X = self._convert_X(X)

        check_classification_targets(y)

        self.n_instances_, self.n_channels_, self.n_timepoints_ = X.shape
        self.classes_ = np.unique(y)
        self.n_classes_ = self.classes_.shape[0]
        self.class_dictionary_ = {}
        for index, class_val in enumerate(self.classes_):
            self.class_dictionary_[class_val] = index

        if self.n_classes_ == 1:
            return self

        self._transformer = QuantTransform(
            depth=self.depth,
            div=self.div,
        )

        self._estimator = _clone_estimator(
            ExtraTreesClassifier(
                n_estimators=200, max_features=0.1, criterion="entropy"
            )
            if self.estimator is None
            else self.estimator,
            self.random_state,
        )

        X_t = self._transformer.fit_transform(X, y)
        self._estimator.fit(X_t, y)

        return self

    def predict(self, X):
        """Predicts labels for sequences in X.

        Parameters
        ----------
        X : 3D np.array of shape (n_instances, n_channels, n_timepoints)
            The testing data.

        Returns
        -------
        y : array-like of shape (n_instances)
            Predicted class labels.
        """
        check_is_fitted(self)

        # treat case of single class seen in fit
        if self.n_classes_ == 1:
            return np.repeat(list(self.class_dictionary_.keys()), X.shape[0], axis=0)

        X = self._validate_data(X=X, reset=False)
        X = self._convert_X(X)

        return self._estimator.predict(self._transformer.transform(X))

    def predict_proba(self, X):
        """Predicts labels probabilities for sequences in X.

        Parameters
        ----------
        X : 3D np.array of shape (n_instances, n_channels, n_timepoints)
            The testing data.

        Returns
        -------
        y : array-like of shape (n_instances, n_classes_)
            Predicted probabilities using the ordering in classes_.
        """
        check_is_fitted(self)

        # treat case of single class seen in fit
        if self.n_classes_ == 1:
            return np.repeat([[1]], X.shape[0], axis=0)

        X = self._validate_data(X=X, reset=False)
        X = self._convert_X(X)

        m = getattr(self._estimator, "predict_proba", None)
        if callable(m):
            return self._estimator.predict_proba(self._transformer.transform(X))
        else:
            dists = np.zeros((X.shape[0], self.n_classes_))
            preds = self._estimator.predict(self._transformer.transform(X))
            for i in range(0, X.shape[0]):
                dists[i, self.class_dictionary_[preds[i]]] = 1
            return dists


class QuantTransform(TransformerMixin, BaseTimeSeriesEstimator):
    """QUANT transform."""

    def __init__(self, depth=6, div=4):
        self.depth = depth
        self.div = div

        super(QuantTransform, self).__init__()

    def fit(self, X, y=None):
        """Fits transformer to X."""
        X = self._validate_data(X=X, ensure_min_samples=2)
        X = self._convert_X(X)

        if self.div < 1 or self.depth < 1:
            raise ValueError("depth and div must be >= 1")

        self.representation_functions = (
            lambda X: X,
            lambda X: F.avg_pool1d(F.pad(X.diff(), (2, 2), "replicate"), 5, 1),
            lambda X: X.diff(n=2),
            lambda X: torch.fft.rfft(X).abs(),
        )

        self.intervals = []
        for function in self.representation_functions:
            Z = function(X)
            self.intervals.append(
                self._make_intervals(
                    input_length=Z.shape[-1],
                    depth=self.depth,
                )
            )

        return self

    def transform(self, X):
        """Transform X."""
        check_is_fitted(self)

        X = self._validate_data(X=X, reset=False)
        X = self._convert_X(X)

        Xt = []
        for index, function in enumerate(self.representation_functions):
            Z = function(X)
            features = []
            for a, b in self.intervals[index]:
                features.append(self._f_quantile(Z[..., a:b], div=self.div).squeeze(1))
            Xt.append(torch.cat(features, -1))

        return torch.cat(Xt, -1)

    @staticmethod
    def _make_intervals(input_length, depth):
        exponent = min(depth, int(np.log2(input_length)) + 1)
        intervals = []
        for n in 2 ** torch.arange(exponent):
            indices = torch.linspace(0, input_length, n + 1).long()
            intervals_n = torch.stack((indices[:-1], indices[1:]), 1)
            intervals.append(intervals_n)
            if n > 1 and intervals_n.diff().median() > 1:
                shift = int(np.ceil(input_length / n / 2))
                intervals.append((intervals_n[:-1] + shift))
        return torch.cat(intervals)

    @staticmethod
    def _f_quantile(X, div=4):
        n = X.shape[-1]

        if n == 1:
            return X
        else:
            num_quantiles = 1 + (n - 1) // div
            if num_quantiles == 1:
                return X.quantile(torch.tensor([0.5]), dim=-1).permute(1, 2, 0)
            else:
                quantiles = X.quantile(
                    torch.linspace(0, 1, num_quantiles), dim=-1
                ).permute(1, 2, 0)
                quantiles[..., 1::2] = quantiles[..., 1::2] - X.mean(-1, keepdims=True)
                return quantiles
