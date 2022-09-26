
from sktime.datasets import load_basic_motions
from sktime.datasets import load_from_tsfile_to_dataframe
from sktime.utils.sampling import stratified_resample
from sktime.classification.kernel_based import RocketClassifier

def debug_types():
    X, y = load_basic_motions()
    trX, trY = load_from_tsfile_to_dataframe(
        full_file_path_and_name="C:\Code\sktime\sktime\datasets\data\BasicMotions"
                                "\BasicMotions_TRAIN.ts")
    testX, testY = load_from_tsfile_to_dataframe(
        full_file_path_and_name="C:\Code\sktime\sktime\datasets\data\BasicMotions"
                                "\BasicMotions_TEST.ts")

    newTrainX, newTrainY, newTestX,newTestY = stratified_resample(
                    trX, trY, testX, testY, 2)
    from sktime.datatypes import check_is_mtype, MTYPE_LIST_PANEL
    from sktime.datatypes._panel._registry import MTYPE_REGISTER_PANEL
    print(MTYPE_LIST_PANEL)
    print(MTYPE_REGISTER_PANEL)
    from sktime import show_versions; show_versions()
    #cls = RocketClassifier()
    #cls.fit(X,y)
    #cls.fit(trX, trY)#
    #cls.fit(newTrainX, newTrainY)
    #p1 = cls.predict(testX)
    #p2 = cls.predict(newTestX)
    #p3 = cls.predict(X)
    #print(p1.shape)
    #print(p2.shape)
    #print(p3.shape)
    #test_set_classifier()

from sktime.classification.dummy import DummyClassifier
import numpy as np
import pandas as pd

def resample_bug():
    from sktime.utils._testing.panel import _make_panel_X
    dummy = DummyClassifier()
    # this works
    trainX = _make_panel_X(n_instances=40)
    trainY = np.random.randint(low=0,high=2,size=40)
    #dummy.fit(trainX,trainY)
    trainX2 = _make_panel_X(n_instances=40)
    trainX2=pd.concat([trainX2,trainX])
    trainY2 = np.random.randint(low=0,high=2,size=80)
    # this throws a type error
    #dummy.fit(trainX2,trainY2)
    trainX = _make_panel_X(n_instances=40)
    trainY = np.random.randint(low=0,high=2,size=40)
    tsf1.fit(trainX, trainY)

def loading_issue_recreate():
    tsf2 = TimeSeriesForestClassifier()
    print(trainX.shape)
    print(trainX.shape)
    from sktime.utils._testing.panel import _make_panel_X
    from sktime.classification.interval_based import SupervisedTimeSeriesForest
    from sktime.classification.interval_based import TimeSeriesForestClassifier
    tsf1 = SupervisedTimeSeriesForest()
    X = pd.DataFrame()
    for i in range(40):
        data = np.arange(40).reshape(2, 20)
        d = {"i": [pd.Series(data[1,:], copy=False)]}
        df_tmp = pd.DataFrame(data=d)
        X = pd.concat([X, df_tmp], ignore_index = True)
    tsf1.fit(X, trainY)
    p=tsf1.predict(X)
    print(" Shape of one series = ",X.iloc[0][0].shape," type = ", type(X.iloc[0][0]))
    print("Built successfully train preds = ",p)


def japanese_vowels_debug():
    from sktime.datasets import load_from_tsfile
    trainX, trainY = load_from_tsfile(full_file_path_and_name="X:\\ArchiveData\\Multivariate_ts\\JapaneseVowels\\JapaneseVowels_TRAIN.ts", return_data_type="nested_univ")
    print(" Train shape = ", trainX.shape)

    for i in range(0,270):
        print(" shape = ", trainX.iloc[i,0].shape)
    print(" first case first dimension= ", trainX.iloc[0,0].shape, " type = ", type(trainX.iloc[0,0]))
    print(" first case first dimension= ", trainX.iloc[0,0])


japanese_vowels_debug()

