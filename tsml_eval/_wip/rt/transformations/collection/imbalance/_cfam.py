import os
import argparse
import numpy as np
import pandas as pd
import torch
from collections import OrderedDict
from typing import Optional, Union

from sklearn.utils import check_random_state
from tsml_eval._wip.rt.transformations.collection.imbalance.pk_cfamg.cfamg import CFAMG
from tsml_eval._wip.rt.transformations.collection.imbalance.pk_cfamg.main import parse_args
from tsml_eval._wip.rt.transformations.collection.imbalance.pk_cfamg.data_preprocess import set_seed
from aeon.transformations.collection import BaseCollectionTransformer

__all__ = ["CFAM"]


class CFAM(BaseCollectionTransformer):
    _tags = {
        "capability:multivariate": True,
        "capability:unequal_length": False,
        "requires_y": True,
    }

    def __init__(self,
                 random_state=None,
                 ):
        self.random_state = random_state
        self.generated_samples = None
        set_seed(self.random_state)
        super().__init__()

    def _fit(self, X, y=None):
        args = parse_args()
        args.w_lambda, args.w_beta = 1, 1
        unique, counts = np.unique(y, return_counts=True)
        target_stats = dict(zip(unique, counts))
        class_majority = max(target_stats, key=target_stats.get)
        class_minority = min(target_stats, key=target_stats.get)
        X_majority, X_minority = X[y == class_majority], X[y == class_minority]
        y_majority, y_minority = y[y == class_majority], y[y == class_minority]

        class_label_project = {class_majority: 0, class_minority: 1}
        self.class_label_project = class_label_project
        y = np.array([class_label_project[label] for label in y])
        y_majority = np.array([class_label_project[label] for label in y_majority])
        y_minority = np.array([class_label_project[label] for label in y_minority])

        ir = len(y_majority) / len(y_minority)
        dataset = {
            "train_data": (X, y),
            "train_data_pos": (X_minority, y_minority),
            "train_data_neg": (X_majority, y_majority),
            "ir": ir,
        }
        args.dataset = dataset
        self.CFAMG_model = CFAMG(args)
        self.CFAMG_model.train_on_data()
        return self

    def _transform(self, X, y=None):
        X_train, y_train, generated_samples = self.CFAMG_model.generator_sample()
        inv_class_label_project = {v: k for k, v in self.class_label_project.items()}
        y_train = np.array([inv_class_label_project[label] for label in y_train])
        self.generated_samples = generated_samples
        return X_train, y_train


if __name__ == "__main__":

    n_samples = 100  # Total number of labels
    majority_num = 90  # number of majority class
    minority_num = n_samples - majority_num  # number of minority class
    np.random.seed(42)

    X = np.random.rand(n_samples, 1, 10)
    y = np.array([0] * majority_num + [1] * minority_num)
    print(np.unique(y, return_counts=True))
    smote = CFAM(random_state=42)

    X_resampled, y_resampled = smote.fit_transform(X, y)
    print(X_resampled.shape)
    print(np.unique(y_resampled, return_counts=True))
