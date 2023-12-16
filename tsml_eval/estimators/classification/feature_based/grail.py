"""GRAIL classifier."""

import os
import sys

import numpy as np
from sklearn.base import ClassifierMixin
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from sklearn.utils.multiclass import check_classification_targets
from sklearn.utils.validation import check_is_fitted
from tsml.base import BaseTimeSeriesEstimator
from tsml.utils.validation import _check_optional_dependency


class GRAIL(ClassifierMixin, BaseTimeSeriesEstimator):
    """
    GRAIL classifier.

    Examples
    --------
    >>> from tsml.datasets import load_minimal_chinatown
    >>> from tsml_eval.estimators.classification.feature_based.grail import GRAIL
    >>> X, y = load_minimal_chinatown()
    >>> clf = GRAIL()
    >>> clf.fit(X, y)
    GRAIL(...)
    >>> preds = clf.predict(X)
    """

    def __init__(self, classifier="svm"):
        self.classifier = classifier

        _check_optional_dependency("grailts", "GRAIL", self)

        super(GRAIL, self).__init__()

    def fit(self, X, y):
        """Fit the estimator to training data.

        Parameters
        ----------
        X : 2D np.ndarray of shape (n_instances, n_timepoints)
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

        self.n_instances_, self.n_timepoints_ = X.shape
        self.classes_ = np.unique(y)
        self.n_classes_ = self.classes_.shape[0]
        self.class_dictionary_ = {}
        for index, class_val in enumerate(self.classes_):
            self.class_dictionary_[class_val] = index

        if self.n_classes_ == 1:
            return self

        self._d = int(self.n_instances_ * 0.4)
        if self._d > 100:
            self._d = 100
        elif self._d < 3:
            self._d = 3

        (
            Xt,
            self._Dictionary,
            self._gamma,
            self._eigenvecMatrix,
            self._inVa,
        ) = self._modified_GRAIL_rep_fit(X, self._d)

        if self.classifier == "svm":
            self._clf = GridSearchCV(
                SVC(kernel="linear", probability=True),
                param_grid={"C": [i**2 for i in np.arange(-10, 20, 0.11)]},
                cv=5,
            )
            self._clf.fit(Xt, y)
        elif self.classifier == "knn":
            self._train_Xt = Xt
            self._train_y = y
        else:
            raise ValueError("classifier must be 'svm' or 'knn'")

        return self

    def predict(self, X):
        """Predicts labels for sequences in X.

        Parameters
        ----------
        X : 2D np.array of shape (n_instances, n_timepoints)
            The testing data.

        Returns
        -------
        y : array-like of shape (n_instances)
            Predicted class labels.
        """
        return np.array(
            [self.classes_[int(np.argmax(prob))] for prob in self.predict_proba(X)]
        )

    def predict_proba(self, X):
        """Predicts labels probabilities for sequences in X.

        Parameters
        ----------
        X : 2D np.array of shape (n_instances, n_timepoints)
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

        Xt = self._modified_GRAIL_rep_predict(
            X, self._d, self._Dictionary, self._gamma, self._eigenvecMatrix, self._inVa
        )

        if self.classifier == "svm":
            probas = self._clf.predict_proba(Xt)
        elif self.classifier == "knn":
            from GRAIL.kNN import kNN

            k = 5
            neighbors, _, _ = kNN(
                self._train_Xt,
                Xt,
                method="ED",
                k=k,
                representation=None,
                pq_method="opq",
            )

            probas = np.zeros((len(X), self.n_classes_))
            for i, case in enumerate(neighbors):
                for j in range(k):
                    probas[i, self.class_dictionary_[self._train_y[case[j]]]] += 1
                probas[i] /= k
        else:
            raise ValueError("classifier must be 'svm' or 'knn'")

        return probas

    @staticmethod
    def _modified_GRAIL_rep_fit(
        X,
        d,
        r=20,
        GV=None,
        fourier_coeff=-1,
        e=-1,
        eigenvecMatrix=None,
        inVa=None,
        gamma=None,
        initialization_method="k-shape++",
    ):
        """Fit GRAIL representation.

        A modified version of the GRAIL_rep function from GRAIL.
        """
        from GRAIL import exceptions
        from GRAIL.GRAIL_core import CheckNaNInfComplex, gamma_select
        from GRAIL.kshape import kshape_with_centroid_initialize, matlab_kshape
        from GRAIL.SINK import SINK

        old_stdout = sys.stdout
        sys.stdout = open(os.devnull, "w")

        n = X.shape[0]
        if initialization_method == "partition":
            [_, Dictionary] = matlab_kshape(X, d)
        elif initialization_method == "centroid_uniform":
            [_, Dictionary] = kshape_with_centroid_initialize(X, d, is_pp=False)
        elif initialization_method == "k-shape++":
            [_, Dictionary] = kshape_with_centroid_initialize(X, d, is_pp=True)
        else:
            raise exceptions.InitializationMethodNotFound

        sys.stdout = old_stdout

        if gamma is None:
            if GV is None:
                GV = [*range(1, 21)]

            [_, gamma] = gamma_select(Dictionary, GV, r)

        E = np.zeros((n, d))
        for i in range(n):
            for j in range(d):
                E[i, j] = SINK(X[i, :], Dictionary[j, :], gamma, fourier_coeff, e)

        if eigenvecMatrix is None and inVa is None:
            W = np.zeros((d, d))
            for i in range(d):
                for j in range(d):
                    W[i, j] = SINK(
                        Dictionary[i, :], Dictionary[j, :], gamma, fourier_coeff, e
                    )

            [eigenvalvector, eigenvecMatrix] = np.linalg.eigh(W)
            inVa = np.diag(np.power(eigenvalvector, -0.5))

        Zexact = E @ eigenvecMatrix @ inVa
        Zexact = CheckNaNInfComplex(Zexact)
        Zexact = np.real(Zexact)

        return Zexact, Dictionary, gamma, eigenvecMatrix, inVa

    @staticmethod
    def _modified_GRAIL_rep_predict(
        X, d, Dictionary, gamma, eigenvecMatrix, inVa, f=0.99, fourier_coeff=-1, e=-1
    ):
        """Predict GRAIL representation.

        A modified version of the GRAIL_rep function from GRAIL.
        """
        from GRAIL.GRAIL_core import CheckNaNInfComplex
        from GRAIL.SINK import SINK

        n = X.shape[0]
        E = np.zeros((n, d))
        for i in range(n):
            for j in range(d):
                E[i, j] = SINK(X[i, :], Dictionary[j, :], gamma, fourier_coeff, e)

        Zexact = E @ eigenvecMatrix @ inVa
        Zexact = CheckNaNInfComplex(Zexact)
        Zexact = np.real(Zexact)

        return Zexact

    def _more_tags(self) -> dict:
        return {
            "X_types": ["2darray"],
            "optional_dependency": True,
            "univariate_only": True,
            "non_deterministic": True,
        }
