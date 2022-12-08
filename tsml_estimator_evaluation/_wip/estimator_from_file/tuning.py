# -*- coding: utf-8 -*-
"""TODO"""

__author__ = ["MatthewMiddlehurst", "ander-hg"]

import numpy as np
from sktime.datasets import load_italy_power_demand
from sklearn.model_selection import GridSearchCV
from tsml_estimator_evaluation._wip.estimator_from_file.hivecote import FromFileHIVECOTE

def tuning_hivecote_alpha_value():
    """TODO"""
    train_X, train_y = load_italy_power_demand(split="train")
    test_X, test_y = load_italy_power_demand(split="test")

    # C:/Users/Ander/git/tsml-estimator-evaluation/tsml_estimator_evaluation/_wip/estimator_from_file/tests/test_files/Arsenal/
    file_paths = [
        "tests/test_files/Arsenal/",
        "tests/test_files/DrCIF/",
        "tests/test_files/STC/",
        "tests/test_files/TDE/",
    ]
    print("oi")
    # hyperparameters
    parameters = {
        'alpha': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        'file_paths': file_paths,
        "random_state": 0
    }

    acc_cv = []

    for i in parameters['alpha']:
        hc2 = FromFileHIVECOTE(file_paths=file_paths, random_state=0, alpha=i)
        hc2.fit(train_X, train_y)
        acc_cv.append(hc2.predict_proba(test_X))

    print(acc_cv)


    """"
#file_paths=file_paths, random_state=0
    model = FromFileHIVECOTE()

    # grid search
    classifier = GridSearchCV(model, parameters, cv=5)

    # fitting the data to our model
    classifier.fit(X, y)

    best_parameters = classifier.best_params_
    print(best_parameters)

    highest_accuracy = classifier.best_score_
    print(highest_accuracy)


"""

if __name__ == '__main__':
   tuning_hivecote_alpha_value()