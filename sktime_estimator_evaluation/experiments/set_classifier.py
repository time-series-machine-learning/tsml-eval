# -*- coding: utf-8 -*-
"""Set classifier function."""
__author__ = ["TonyBagnall"]

from sklearn.ensemble import RandomForestClassifier

from sktime.classification.deep_learning import CNNClassifier
#from sktime.classification.dummy import DummyClassifier
from sktime.classification.dictionary_based import (
    MUSE,
    WEASEL,
    BOSSEnsemble,
    ContractableBOSS,
    TemporalDictionaryEnsemble,
    IndividualTDE,
    IndividualBOSS,
)
from sktime.classification.distance_based import (
    ElasticEnsemble,
    KNeighborsTimeSeriesClassifier,
    ProximityForest,
    ProximityStump,
    ProximityTree,
    ShapeDTW,
)
from sktime.classification.feature_based import (
    Catch22Classifier,
    FreshPRINCE,
    MatrixProfileClassifier,
    RandomIntervalClassifier,
    SignatureClassifier,
    SummaryClassifier,
    TSFreshClassifier,
)
from sktime.classification.hybrid import HIVECOTEV1, HIVECOTEV2
from sktime.classification.interval_based import (
    CanonicalIntervalForest,
    DrCIF,
    RandomIntervalSpectralEnsemble,
    SupervisedTimeSeriesForest,
    TimeSeriesForestClassifier,
)
from sktime.classification.kernel_based import Arsenal, RocketClassifier
from sktime.classification.shapelet_based import ShapeletTransformClassifier
from sktime.classification.compose import ComposableTimeSeriesForestClassifier
from sktime.transformations.series.summarize import SummaryTransformer


multivariate_classifiers = [
"Arsenal",
"CNNClassifier",
"CanonicalIntervalForest",
"Catch22Classifier",
"ColumnEnsembleClassifier",
"DrCIF",
"FreshPRINCE",
"HIVECOTEV2",
"IndividualTDE",
"KNeighborsTimeSeriesClassifier",
"MUSE",
"ProbabilityThresholdEarlyClassifier",
"RandomIntervalClassifier",
"RocketClassifier",
"ShapeletTransformClassifier",
"SignatureClassifier",
"SummaryClassifier",
"TSFreshClassifier",
"TemporalDictionaryEnsemble",
"WeightedEnsembleClassifier",
]

def set_classifier(cls, resample_id=None, train_file=False):
    """Construct a classifier, possibly seeded.

    Basic way of creating the classifier to build using the default settings. This
    set up is to help with batch jobs for multiple problems to facilitate easy
    reproducibility for use with load_and_run_classification_experiment. You can pass a
    classifier object instead to run_classification_experiment.
    TODO: add contract, checkpoint and threaded options

    Parameters
    ----------
    cls : str
        String indicating which classifier you want.
    resample_id : int or None, default=None
        Classifier random seed.
    train_file : bool, default=False
        Whether a train file is being produced.

    Return
    ------
    classifier : A BaseClassifier.
        The classifier matching the input classifier name.
    """
    name = cls.lower()
    # Dictionary based
    if name == "boss" or name == "bossensemble":
        return BOSSEnsemble(random_state=resample_id)
    elif name == "individualboss":
        return IndividualBOSS(
            random_state=resample_id
        )
    elif name == "cboss" or name == "contractableboss":
        return ContractableBOSS(random_state=resample_id)
    elif name == "tde" or name == "temporaldictionaryensemble":
        return TemporalDictionaryEnsemble(random_state=resample_id, save_train_predictions=train_file)
    elif name == "individualtde":
        return IndividualTDE(random_state=resample_id)
    elif name == "weasel":
        return WEASEL(random_state=resample_id)
    elif name == "muse":
        return MUSE(random_state=resample_id)
    # Distance based
    elif name == "pf" or name == "proximityforest":
        return ProximityForest(random_state=resample_id)
    elif name == "pt" or name == "proximitytree":
        return ProximityTree(random_state=resample_id)
    elif name == "ps" or name == "proximitystump":
        return ProximityStump(random_state=resample_id)
    elif name == "dtw" or name == "1nn-dtw" or name == "KNeighborsTimeSeriesClassifier":
        return KNeighborsTimeSeriesClassifier()
    elif name == "ed" or name == "1nn-ed":
        return KNeighborsTimeSeriesClassifier(distance="ed")
    elif name == "msm" or name == "1nn-msm":
        return KNeighborsTimeSeriesClassifier(distance="msm")
    elif name == "ee" or name == "elasticensemble":
        return ElasticEnsemble(random_state=resample_id)
    elif name == "shapedtw":
        return ShapeDTW()
    # Feature based
    elif name == "summary" or name == "summaryclassifier":
        return SummaryClassifier(
            random_state=resample_id, estimator=RandomForestClassifier(n_estimators=500)
        )
    elif name == "summary-intervals":
        return RandomIntervalClassifier(
            random_state=resample_id,
            interval_transformers=SummaryTransformer(
                summary_function=("mean", "std", "min", "max"),
                quantiles=(0.25, 0.5, 0.75),
            ),
            estimator=RandomForestClassifier(n_estimators=500),
        )
    elif name == "summary-catch22":
        return RandomIntervalClassifier(
            random_state=resample_id, estimator=RandomForestClassifier(n_estimators=500)
        )
    elif name == "catch22" or name == "catch22classifier":
        return Catch22Classifier(
            random_state=resample_id, estimator=RandomForestClassifier(n_estimators=500)
        )
    elif name == "matrixprofile" or name == "matrixprofileclassifier":
        return MatrixProfileClassifier(random_state=resample_id)
    elif name == "freshprince":
        return FreshPRINCE(random_state=resample_id)
    elif name == "randomintervalclassifier" or name == "randominterval":
        return RandomIntervalClassifier(random_state=resample_id)
    elif name == "tsfreshclassifier" or  name == "tsfresh":
        return TSFreshClassifier(random_state=resample_id)
    # hybrids
    elif name == "hc1" or name == "hivecotev1":
        return HIVECOTEV1(random_state=resample_id)
    elif name == "hc2" or name == "hivecotev2":
        return HIVECOTEV2(random_state=resample_id)

    # Interval based
    elif name == "rise" or name == "randomintervalspectralforest" or name == "randomintervalspectralensemble":
        return RandomIntervalSpectralEnsemble(
            random_state=resample_id, n_estimators=500
        )
    elif name == "tsf" or name == "timeseriesforestclassifier":
        return TimeSeriesForestClassifier(random_state=resample_id, n_estimators=500)
    elif name == "cif" or name == "canonicalintervalforest":
        return CanonicalIntervalForest(random_state=resample_id, n_estimators=500)
    elif name == "stsf" or name == "supervisedtimeseriesforest":
        return SupervisedTimeSeriesForest(random_state=resample_id, n_estimators=500)
    elif name == "drcif":
        return DrCIF(
            random_state=resample_id, n_estimators=500, save_transformed_data=train_file
        )
    # Convolution based
    elif name == "rocket" or name == "rocketclassifier":
        return RocketClassifier(random_state=resample_id)
    elif name == "mini-rocket":
        return RocketClassifier(random_state=resample_id, rocket_transform="minirocket")
    elif name == "multi-rocket":
        return RocketClassifier(
            random_state=resample_id, rocket_transform="multirocket"
        )
    elif name == "arsenal":
        return Arsenal(random_state=resample_id, save_transformed_data=train_file)
    elif name == "mini-arsenal":
        return Arsenal(
            random_state=resample_id,
            save_transformed_data=train_file,
            rocket_transform="minirocket",
        )
    elif name == "multi-arsenal":
        return Arsenal(
            random_state=resample_id,
            save_transformed_data=train_file,
            rocket_transform="multirocket",
        )
    # Shapelet based
    elif name == "stc" or name == "shapelettransformclassifier":
        return ShapeletTransformClassifier(
            transform_limit_in_minutes=120,
            random_state=resample_id,
            save_transformed_data=train_file,
        )
    elif name == "composabletimeseriesforestclassifier":
        return ComposableTimeSeriesForestClassifier()
#    elif name == "dummy" or name == "dummyclassifier":
 #       return DummyClassifier()
    # deep learning based
    elif name == "signatureclassifier":
        return SignatureClassifier(random_state=resample_id)
#        raise Exception("Need the soft dep esig package for signatures")
    elif name == "cnn" or name == "cnnclassifier":
        return CNNClassifier()
    # requires constructor arguments
    elif name == "columnensemble" or name == "columnensembleclassifier":
        raise Exception("Cannot create a ColumnEnsembleClassifier without passing a base "
              "classifier ")
    elif name == "probabilitythresholdearlyclassifier":
        raise Exception("probabilitythresholdearlyclassifier is for early classification, "
              "not applicable here")
    elif name == "classifierpipeline" or name == "sklearnclassifierpipeline":
        raise Exception("Cannot create a ClassifierPipeline or SklearnClassifierPipeline "
              "without passing a base "
              "classifier and transform(s)")
    elif name == "weightedensembleclassifier" or name == "weightedensemble":
        raise Exception("Cannot create a WeightedEnsembleClassifier"
              "without passing base classifiers")
    else:
        raise Exception("UNKNOWN CLASSIFIER ", name," in set_classifier")

