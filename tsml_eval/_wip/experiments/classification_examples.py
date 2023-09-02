# -*- coding: utf-8 -*-
# Simple usages for building classifiers

import numpy as np
from aeon.classification.compose import ClassifierPipeline
from aeon.classification.feature_based import FreshPRINCE
from aeon.datasets import load_arrow_head, load_UCR_UEA_dataset
from aeon.transformations.panel.catch22 import Catch22

from tsml_eval.evaluation import fetch_classifier_metric
from tsml_eval.evaluation.diagrams import critical_difference_diagram, scatter_diagram


def basic_usage():
    arrow_X, arrow_y = load_arrow_head(return_type="numpy2d")
    # work directly with numpy 2D for equal length univariate
    # 10 series equal length of 50
    train_X = np.random.rand(10, 50)
    # 20 series equal length of 50
    test_X = np.random.rand(20, 50)
    # Random class labels
    train_Y = np.random.randint(0, high=2, size=10)
    test_Y = np.random.randint(0, high=2, size=20)
    freshPrince = FreshPRINCE()
    freshPrince.fit(train_X, train_Y)
    preds = freshPrince.predict(test_X)
    print("Univariate preds", preds)
    # work directly with numpy 3D for equal length univariate
    # series of 3 dimensions, equal length of 50
    train_X = np.random.rand(10, 3, 50)
    test_X = np.random.rand(20, 3, 50)
    freshPrince.fit(train_X, train_Y)
    preds = freshPrince.predict(test_X)
    print(" Multivariate preds ", preds)
    # Load default train/test splits from sktime/datasets/data
    arrow_train_X, arrow_train_y = load_arrow_head(split="train", return_type="numpy2d")
    arrow_test_X, arrow_test_y = load_arrow_head(split="test", return_type="numpy2d")
    freshPrince.fit(arrow_train_X, arrow_train_y)
    s = freshPrince.score(arrow_test_X, arrow_test_y)
    print(" Score  = ", s)


def ucr_datasets(classifier, examples):
    scores = np.zeros(len(examples))
    for i in range(0, len(examples)):
        train_X, train_y = load_UCR_UEA_dataset(examples[i], split="TRAIN")
        test_X, test_y = load_UCR_UEA_dataset(examples[i], split="TEST")
        classifier.fit(train_X, train_y)
        scores[i] = classifier.score(test_X, test_y)
        print(" problem ", examples[i], " accuracy = ", scores[i])
    return scores


examples = ["Chinatown", "ItalyPowerDemand", "BeetleFly", "Adiac"]
# freshPrince = FreshPRINCE()
# acc = ucr_datasets(freshPrince, examples)
results = [0.9, 0.8, 0.7, 0.6]
names = ["FreshPrince"]
others = ["HC2", "InceptionTime", "ROCKET"]
other_accs_df = fetch_classifier_metric(
    metrics=["ACC"],
    datasets=examples,
    classifiers=others,
    folds=1,
)
for res in zip(examples, results):
    other_accs_df.loc[len(other_accs_df)] = ["FreshPrince", res[0], res[1]]
cd = critical_difference_diagram(other_accs_df)
scatters = scatter_diagram(
    other_accs_df,
    compare_estimators_from=["FreshPrince"],
)

cd.show()

for scatter in scatters:
    scatter.show()

# Combine results and other_accs into one CD

# print as a table

# Make your own Pipeline
# summarytStats = Catch22
# ClassifierPipeline p =
