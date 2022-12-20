# -*- coding: utf-8 -*-
from sktime.datasets import load_arrow_head, load_unit_test

from tsml_eval.utils.experiments import resample


def test_resample():
    train_X, train_y = load_arrow_head(split="train")
    test_X, test_y = load_unit_test(split="train")

    train_size = train_y.size
    test_size = test_y.size

    train_X, train_y, test_X, test_y = resample(train_X, train_y, test_X, test_y, 1)

    assert train_y.size == train_size and test_y.size == test_size

    
def test_stratified_resample():
    """Test resampling returns valid data structure and maintains class distribution."""
    trainX, trainy = load_unit_test(split="TRAIN")
    testX, testy = load_unit_test(split="TEST")
    new_trainX, new_trainy, new_testX, new_testy = stratified_resample(
        trainX, trainy, testX, testy, 0
    )

    valid_train = check_is_scitype(new_trainX, scitype="Panel")
    valid_test = check_is_scitype(new_testX, scitype="Panel")
    assert valid_test and valid_train
    # count class occurrences
    unique_train, counts_train = np.unique(trainy, return_counts=True)
    unique_test, counts_test = np.unique(testy, return_counts=True)
    unique_train_new, counts_train_new = np.unique(new_trainy, return_counts=True)
    unique_test_new, counts_test_new = np.unique(new_testy, return_counts=True)
    assert list(counts_train_new) == list(counts_train)
    assert list(counts_test_new) == list(counts_test)
