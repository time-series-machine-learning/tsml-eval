import numpy as np
from sklearn.linear_model import RidgeClassifierCV
rng = np.random.RandomState(0)
X = rng.normal(size=(200, 20))
y = rng.randint(0, 2, size=200)

clf_loo = RidgeClassifierCV( alphas=[0.1, 1.0, 10.0], scoring="accuracy", ).fit(X, y)
clf_cv = RidgeClassifierCV( alphas=[0.1, 1.0, 10.0], scoring="accuracy", cv=5, ).fit(X, y)
print("cv=None best_score_:", clf_loo.best_score_)
print("cv=5 best_score_:", clf_cv.best_score_)
print("training accuracy:", clf_loo.score(X, y))

import sklearn;
sklearn.show_versions()
