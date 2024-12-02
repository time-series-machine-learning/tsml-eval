"""tsml-eval imbalance classification utils using smote and its variants to rebalanced data."""

__author__ = ["Chris Qiu"]

__all__ = [
    "SMOTE_FAMILY",
]

from imblearn.over_sampling import ADASYN, RandomOverSampler, SMOTE

class SMOTE_FAMILY:
    """
    over-sampling methods include 'ADASYN', 'RandomOverSampler', 'KMeansSMOTE', 'SMOTE',
    'BorderlineSMOTE', 'SVMSMOTE', 'SMOTENC', 'SMOTEN'
    """

    def ros(self, seed):
        return RandomOverSampler(random_state=seed)

    def rose(self, seed):
        return RandomOverSampler(random_state=seed, shrinkage={1: 2.0})

    def adasyn(self, seed):
        return ADASYN(random_state=seed, n_neighbors=3)

    def smote(self, seed):
        return SMOTE(random_state=seed, k_neighbors=3)

    # def kmeans_smote(self, seed):
    #     return KMeansSMOTE(random_state=seed)
    #
    # def borderline_smote(self, seed):
    #     return BorderlineSMOTE(random_state=seed, kind="borderline-1")
