from itertools import product

import numpy as np
from sklearn.utils.validation import check_random_state
from aeon.classification.dictionary_based._boss import boss_distance
from aeon.transformations.collection.dictionary_based import SFAFast


class DictionarySplitter:
    """BOSS-based splitter for TS-CHIEF implementation."""

    @staticmethod
    def generate(X_boss, y, sfas, random_state=None):
        """Generate a randomized dictionary splitter candidate."""
        dims, num_transforms = X_boss.shape
        rng = check_random_state(random_state)

        splitter = DictionarySplitter()
        splitter.dim = rng.randint(dims)
        splitter.transform_idx = rng.randint(num_transforms)
        splitter.sfa = sfas[splitter.dim, splitter.transform_idx]

        splitter.exemplars = []
        classes = np.unique(y)
        for c in classes:
            group = np.argwhere(y == c).ravel()
            exemplar_idx = rng.choice(group)
            splitter.exemplars.append(
                X_boss[splitter.dim, splitter.transform_idx][exemplar_idx, :]
            )

        return splitter

    def split_train(self, X_boss):
        """Split the training data without needlessly SFAing again."""
        X_boss = X_boss[self.dim, self.transform_idx]
        samples, _ = X_boss.shape

        split_idx = np.empty(samples, dtype=int)
        for i in range(samples):
            distances = [
                boss_distance(X_boss[i, :], exemplar, 0) for exemplar in self.exemplars
            ]
            split_idx[i] = np.argmin(distances)

        return split_idx

    def split(self, X):
        """Split incoming data."""
        X = X[:, [self.dim], :]
        X_boss = self.sfa.transform(X)
        samples, _ = X_boss.shape

        split_idx = np.empty(samples, dtype=int)
        for i in range(samples):
            distances = [
                boss_distance(X_boss[i, :], exemplar, 0) for exemplar in self.exemplars
            ]
            split_idx[i] = np.argmin(distances)

        return split_idx


def generate_boss_transforms(X, num_transforms_per_dim=1000, random_state=None):
    """Generate random SFA transformations for splitters."""
    rng = check_random_state(random_state)
    _, dims, length = X.shape

    window_sizes = np.arange(10, length + 1)
    word_lengths = np.array([6, 8, 10, 12, 14, 16])
    normalizations = np.array([True, False])
    alphabet_sizes = np.array([4])

    params = list(product(window_sizes, word_lengths, normalizations, alphabet_sizes))
    params = np.array(params, dtype=tuple)
    num_transforms_per_dim = min(num_transforms_per_dim, len(params))

    sfas = []
    bags = []

    for dim in range(dims):
        sfas_dim = []
        bags_dim = []

        used_params_idx = rng.choice(
            len(params), size=num_transforms_per_dim, replace=False
        )
        used_params = params[used_params_idx]
        for window_size, word_length, norm, alphabet_size in used_params:
            sfa = SFAFast(
                word_length=word_length,
                alphabet_size=alphabet_size,
                window_size=window_size,
                norm=norm,
                binning_method="equi-depth",
                anova=False,
                variance=False,
                bigrams=False,
                skip_grams=False,
                remove_repeat_words=True,
                feature_selection="none",
                random_state=rng,
                n_jobs=1,
            )
            bag = sfa.fit_transform(X[:, [dim], :])

            sfas_dim.append(sfa)
            bags_dim.append(bag)

        sfas.append(sfas_dim)
        bags.append(bags_dim)

    return np.array(sfas), np.array(bags)
