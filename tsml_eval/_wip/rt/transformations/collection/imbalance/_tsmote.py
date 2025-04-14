"""TSMOTE (Temporal SMOTE) for time series data.

Based on: T-SMOTE: Temporal-oriented Synthetic Minority Oversampling Technique
for Imbalanced Time Series Classification.
"""

from typing import Optional, Union

import numpy as np
from sklearn.utils import check_random_state

from tsml_eval._wip.rt.classification.distance_based import KNeighborsTimeSeriesClassifier
from aeon.transformations.collection import BaseCollectionTransformer


class TSMOTE(BaseCollectionTransformer):
    """Temporal Synthetic Minority Over-sampling TEchnique (T-SMOTE).

    An adaptation of SMOTE specifically for time series data that uses temporal
    information to generate synthetic samples near class borders.

    Parameters
    ----------
    random_state : int, RandomState instance or None, default=None
        If int, random_state is the seed used by the random number generator;
        If RandomState instance, random_state is the random number generator;
        If None, the random number generator is the RandomState instance used
        by np.random.
    spy_size : float, default=0.15
        The proportion of majority samples to use as spies.
    window_size : int, default=None
        The size of the window to use for subsequences. If None, uses the
        full series length.
    distance : str or callable, default='euclidean'
        Distance metric to use for KNN classification of spy samples.
    """

    _tags = {
        "capability:multivariate": True,
        "capability:unequal_length": False,
        "requires_y": True,
    }

    def __init__(
        self,
        random_state=None,
        spy_size: float = 0.15,
        window_size: Optional[int] = None,
        distance: Union[str, callable] = "euclidean",
        distance_params: Optional[dict] = None,
    ):
        self.random_state = random_state
        self.spy_size = spy_size
        self.window_size = window_size
        self.distance = distance
        self.distance_params = distance_params

        self._random_state = None
        self._distance_params = distance_params or {}

        super().__init__()

    def _fit(self, X, y=None):
        """Fit the T-SMOTE transformer."""
        self._random_state = check_random_state(self.random_state)

        # Get minority and majority class indices
        unique, counts = np.unique(y, return_counts=True)
        if len(unique) != 2:
            raise ValueError("T-SMOTE only works for binary classification problems")

        self.minority_class_ = unique[counts.argmin()]
        self.majority_class_ = unique[counts.argmax()]

        minority_idx = np.where(y == self.minority_class_)[0]
        majority_idx = np.where(y == self.majority_class_)[0]

        self.X_minority_ = X[minority_idx]
        self.X_majority_ = X[majority_idx]

        self.n_minority_ = len(minority_idx)
        self.n_majority_ = len(majority_idx)

        # Set window size if not provided
        if self.window_size is None:
            self.window_size_ = X.shape[-1]
        else:
            self.window_size_ = min(self.window_size, X.shape[-1])

        # Initialize classifier for spy-based threshold
        self.classifier_ = KNeighborsTimeSeriesClassifier(
            distance=self.distance,
            distance_params=self._distance_params,
        )

        return self

    def _transform(self, X, y=None):
        """Transform the data using T-SMOTE."""
        # Initialize lists for resampled data
        X_resampled = [X.copy()]
        y_resampled = [y.copy()]

        n_synthetic_needed = self.n_majority_ - self.n_minority_

        # Get majority samples to use as spies
        n_spies = max(1, int(self.n_majority_ * self.spy_size))
        spy_idx = self._random_state.choice(
            self.n_majority_, size=n_spies, replace=False
        )
        X_spies = self.X_majority_[spy_idx]

        # Train classifier on minority + spy samples
        X_spy_train = np.vstack([self.X_minority_, X_spies])
        y_spy_train = np.hstack([np.ones(self.n_minority_), np.zeros(len(X_spies))])
        self.classifier_.fit(X_spy_train, y_spy_train)

        # Get threshold from spy predictions
        spy_scores = self.classifier_.predict_proba(X_spies)[:, 1]
        threshold = np.max(spy_scores)

        # Generate temporal samples for each minority instance
        synthetic_samples = []
        synthetic_scores = []

        for i in range(self.n_minority_):
            # Generate samples with different leading times
            leading_time = 0
            temporal_samples = []
            temporal_scores = []

            while True:
                sample = self._generate_temporal_sample(
                    self.X_minority_[i], leading_time
                )

                if sample is None:
                    break

                score = self.classifier_.predict_proba(
                    sample.reshape(1, *sample.shape)
                )[0, 1]

                if score < threshold:
                    break

                temporal_samples.append(sample)
                temporal_scores.append(score)
                leading_time += 1

                if leading_time >= self.X_minority_[i].shape[-1] - self.window_size_:
                    break

            if len(temporal_samples) < 2:
                continue

            # Convert to arrays
            temporal_samples = np.array(temporal_samples)
            temporal_scores = np.array(temporal_scores)

            # Generate synthetic samples using temporal neighbors
            for j in range(len(temporal_samples) - 1):
                current = temporal_samples[j]
                neighbor = temporal_samples[j + 1]
                current_score = temporal_scores[j]
                neighbor_score = temporal_scores[j + 1]

                alpha = self._random_state.beta(
                    max(0.1, current_score), max(0.1, neighbor_score)
                )

                synthetic = current * alpha + neighbor * (1 - alpha)
                synthetic_score = current_score * alpha + neighbor_score * (1 - alpha)

                synthetic_samples.append(synthetic)
                synthetic_scores.append(synthetic_score)

        if len(synthetic_samples) == 0:
            # If no synthetic samples generated, duplicate minority samples
            synthetic_samples = [
                self.X_minority_[self._random_state.randint(self.n_minority_)]
                for _ in range(n_synthetic_needed)
            ]
            synthetic_scores = [1.0] * n_synthetic_needed
        else:
            # Convert to arrays
            synthetic_samples = np.array(synthetic_samples)
            synthetic_scores = np.array(synthetic_scores)

            # Randomly select samples weighted by scores
            weights = synthetic_scores / synthetic_scores.sum()
            selected_idx = self._random_state.choice(
                len(synthetic_samples), size=n_synthetic_needed, p=weights, replace=True
            )
            synthetic_samples = synthetic_samples[selected_idx]

        # Add synthetic samples
        for synthetic in synthetic_samples:
            X_resampled.append(synthetic.reshape(1, *synthetic.shape))
            y_resampled.append(self.minority_class_)

        return (np.vstack(X_resampled), np.hstack(y_resampled))

    def _generate_temporal_sample(self, x, leading_time):
        """Generate a temporal sample with given leading time."""
        if leading_time == 0:
            return x

        start_idx = leading_time
        end_idx = start_idx + self.window_size_

        if end_idx > x.shape[-1]:
            return None

        # Create new array with same shape as input but different timepoints
        sample = np.zeros_like(x)
        sample[..., : self.window_size_] = x[..., start_idx:end_idx]

        return sample

if __name__ == "__main__":
    # Example usage
    X = np.random.randn(100, 1, 100)
    y = np.random.choice([0, 0, 1], size=100)
    print(np.unique(y, return_counts=True))
    tsmote = TSMOTE(random_state=0)
    X_resampled, y_resampled = tsmote.fit_transform(X, y)
    print(np.unique(y_resampled, return_counts=True))
    # Output: (200, 1, 100) (200,)
