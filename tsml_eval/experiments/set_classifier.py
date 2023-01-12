# -*- coding: utf-8 -*-
"""Set classifier function."""

__author__ = ["TonyBagnall", "MatthewMiddlehurst"]


def set_classifier(
    classifier_name, random_state=None, build_train_file=False, n_jobs=1, fit_contract=0
):
    """Return a classifier matching a given input name.

    Basic way of creating a classifier to build using the default or alternative
    settings. This set up is to help with batch jobs for multiple problems and to
    facilitate easy reproducibility for use with run_classification_experiment.

    Generally, inputting a classifier class name will return said classifier with
    default settings.

    Parameters
    ----------
    classifier_name : str
        String indicating which classifier to be returned.
    random_state : int, RandomState instance or None, default=None
        Random seed or RandomState object to be used in the classifier if available.
    build_train_file : bool, default=False
        Whether a train data results file is being produced. If True, classifier
        specific parameters for generating train results will be toggled if available.
    n_jobs: int, default=1
        The number of jobs to run in parallel for both classifier ``fit`` and
        ``predict`` if available. `-1` means using all processors.
    fit_contract: int, default=0
        Contract time in minutes for classifier ``fit`` if available.

    Return
    ------
    classifier : A BaseClassifier.
        The classifier matching the input classifier name.
    """
    c = classifier_name.lower()

    # Convolution based
    if c == "rocket" or c == "rocketclassifier":
        from sktime.classification.kernel_based import RocketClassifier

        return RocketClassifier(random_state=random_state, n_jobs=n_jobs)
    elif c == "mini-rocket":
        from sktime.classification.kernel_based import RocketClassifier

        return RocketClassifier(
            rocket_transform="minirocket",
            random_state=random_state,
            n_jobs=n_jobs,
        )
    elif c == "multi-rocket":
        from sktime.classification.kernel_based import RocketClassifier

        return RocketClassifier(
            rocket_transform="multirocket",
            random_state=random_state,
            n_jobs=n_jobs,
        )
    elif c == "arsenal":
        from sktime.classification.kernel_based import Arsenal

        return Arsenal(
            random_state=random_state,
            save_transformed_data=build_train_file,
            n_jobs=n_jobs,
            time_limit_in_minutes=fit_contract,
        )
    elif c == "mini-arsenal":
        from sktime.classification.kernel_based import Arsenal

        return Arsenal(
            rocket_transform="minirocket",
            random_state=random_state,
            save_transformed_data=build_train_file,
            n_jobs=n_jobs,
            time_limit_in_minutes=fit_contract,
        )
    elif c == "multi-arsenal":
        from sktime.classification.kernel_based import Arsenal

        return Arsenal(
            rocket_transform="multirocket",
            random_state=random_state,
            save_transformed_data=build_train_file,
            n_jobs=n_jobs,
            time_limit_in_minutes=fit_contract,
        )
    elif c == "hydra":
        from tsml_eval.sktime_estimators.classification.convolution_based.hydra import (
            HYDRA,
        )

        return HYDRA(random_state=random_state)

    # Dictionary based
    if c == "boss" or c == "bossensemble":
        from sktime.classification.dictionary_based import BOSSEnsemble

        return BOSSEnsemble(
            random_state=random_state,
            n_jobs=n_jobs,
            save_train_predictions=build_train_file,
        )
    elif c == "individualboss":
        from sktime.classification.dictionary_based import IndividualBOSS

        return IndividualBOSS(
            random_state=random_state,
            n_jobs=n_jobs,
        )
    elif c == "cboss" or c == "contractableboss":
        from sktime.classification.dictionary_based import ContractableBOSS

        return ContractableBOSS(
            random_state=random_state,
            n_jobs=n_jobs,
            save_train_predictions=build_train_file,
            time_limit_in_minutes=fit_contract,
        )
    elif c == "tde" or c == "temporaldictionaryensemble":
        from sktime.classification.dictionary_based import TemporalDictionaryEnsemble

        return TemporalDictionaryEnsemble(
            random_state=random_state,
            save_train_predictions=build_train_file,
            n_jobs=n_jobs,
            time_limit_in_minutes=fit_contract,
        )
    elif c == "individualtde":
        from sktime.classification.dictionary_based import IndividualTDE

        return IndividualTDE(random_state=random_state, n_jobs=n_jobs)
    elif c == "weasel":
        from sktime.classification.dictionary_based import WEASEL

        return WEASEL(random_state=random_state, n_jobs=n_jobs)
    elif c == "weasel-logistic":
        from sktime.classification.dictionary_based import WEASEL

        return WEASEL(
            random_state=random_state, n_jobs=n_jobs, support_probabilities=True
        )
    elif c == "muse":
        from sktime.classification.dictionary_based import MUSE

        return MUSE(random_state=random_state, n_jobs=n_jobs)
    elif c == "muse-logistic":
        from sktime.classification.dictionary_based import MUSE

        return MUSE(
            random_state=random_state, n_jobs=n_jobs, support_probabilities=True
        )
    elif c == "weasel-dilation":
        from tsml_eval.sktime_estimators.classification.dictionary_based.weasel import (
            WEASEL_DILATION,
        )

        return WEASEL_DILATION(random_state=random_state, n_jobs=n_jobs)
    elif c == "muse-dilation":
        from tsml_eval.sktime_estimators.classification.dictionary_based.muse import (
            MUSE_DILATION,
        )

        return MUSE_DILATION(random_state=random_state, n_jobs=n_jobs)

    # Distance based
    elif c == "pf" or c == "proximityforest":
        from sktime.classification.distance_based import ProximityForest

        return ProximityForest(random_state=random_state, n_jobs=n_jobs)
    elif c == "pt" or c == "proximitytree":
        from sktime.classification.distance_based import ProximityTree

        return ProximityTree(random_state=random_state, n_jobs=n_jobs)
    elif c == "ps" or c == "proximitystump":
        from sktime.classification.distance_based import ProximityStump

        return ProximityStump(random_state=random_state, n_jobs=n_jobs)
    elif c == "dtw" or c == "1nn-dtw" or c == "kneighborstimeseriesclassifier":
        from tsml_eval.sktime_estimators.classification.distance_based import (
            KNeighborsTimeSeriesClassifier,
        )

        return KNeighborsTimeSeriesClassifier(n_jobs=n_jobs)
    elif c == "ed" or c == "1nn-euclidean" or c == "1nn-ed":
        from sktime.classification.distance_based import KNeighborsTimeSeriesClassifier

        return KNeighborsTimeSeriesClassifier(distance="euclidean", n_jobs=n_jobs)
    elif c == "msm" or c == "1nn-msm":
        from sktime.classification.distance_based import KNeighborsTimeSeriesClassifier

        return KNeighborsTimeSeriesClassifier(distance="msm", n_jobs=n_jobs)
    elif c == "ee" or c == "elasticensemble":
        from sktime.classification.distance_based import ElasticEnsemble

        return ElasticEnsemble(random_state=random_state, n_jobs=n_jobs)
    elif c == "shapedtw":
        from sktime.classification.distance_based import ShapeDTW

        return ShapeDTW()

    # Feature based
    elif c == "summary-500":
        from sklearn.ensemble import RandomForestClassifier
        from sktime.classification.feature_based import SummaryClassifier

        return SummaryClassifier(
            estimator=RandomForestClassifier(n_estimators=500),
            random_state=random_state,
            n_jobs=n_jobs,
        )
    elif c == "summaryclassifier" or c == "summary":
        from sktime.classification.feature_based import SummaryClassifier

        return SummaryClassifier(random_state=random_state, n_jobs=n_jobs)
    elif c == "summary-intervals":
        from sklearn.ensemble import RandomForestClassifier
        from sktime.classification.feature_based import RandomIntervalClassifier
        from sktime.transformations.series.summarize import SummaryTransformer

        return RandomIntervalClassifier(
            interval_transformers=SummaryTransformer(
                summary_function=("mean", "std", "min", "max"),
                quantiles=(0.25, 0.5, 0.75),
            ),
            estimator=RandomForestClassifier(n_estimators=500),
            random_state=random_state,
            n_jobs=n_jobs,
        )
    elif c == "randominterval-rf" or c == "catch22intervals-rf":
        from sklearn.ensemble import RandomForestClassifier
        from sktime.classification.feature_based import RandomIntervalClassifier

        return RandomIntervalClassifier(
            estimator=RandomForestClassifier(n_estimators=500),
            random_state=random_state,
            n_jobs=n_jobs,
        )
    elif (
        c == "randomintervalclassifier"
        or c == "randominterval"
        or c == "catch22intervals"
    ):
        from sktime.classification.feature_based import RandomIntervalClassifier

        return RandomIntervalClassifier(random_state=random_state, n_jobs=n_jobs)
    elif c == "catch22-500":
        from sklearn.ensemble import RandomForestClassifier
        from sktime.classification.feature_based import Catch22Classifier

        return Catch22Classifier(
            estimator=RandomForestClassifier(n_estimators=500),
            random_state=random_state,
            n_jobs=n_jobs,
        )
    elif c == "catch22" or c == "catch22classifier":
        from sktime.classification.feature_based import Catch22Classifier

        return Catch22Classifier(random_state=random_state, n_jobs=n_jobs)
    elif c == "matrixprofile" or c == "matrixprofileclassifier":
        from sktime.classification.feature_based import MatrixProfileClassifier

        return MatrixProfileClassifier(random_state=random_state, n_jobs=n_jobs)
    elif c == "freshprince":
        from sktime.classification.feature_based import FreshPRINCE

        return FreshPRINCE(random_state=random_state, n_jobs=n_jobs)
    elif c == "tsfresh-nofs":
        from sktime.classification.feature_based import TSFreshClassifier

        return TSFreshClassifier(
            relevant_feature_extractor=False, random_state=random_state, n_jobs=n_jobs
        )
    elif c == "tsfreshclassifier" or c == "tsfresh":
        from sktime.classification.feature_based import TSFreshClassifier

        return TSFreshClassifier(random_state=random_state, n_jobs=n_jobs)
    elif c == "signatureclassifier" or c == "signatures":
        from sktime.classification.feature_based import SignatureClassifier

        return SignatureClassifier(random_state=random_state)

    # hybrids
    elif c == "hc1" or c == "hivecotev1":
        from sktime.classification.hybrid import HIVECOTEV1

        return HIVECOTEV1(random_state=random_state, n_jobs=n_jobs)
    elif c == "hc2" or c == "hivecotev2":
        from sktime.classification.hybrid import HIVECOTEV2

        return HIVECOTEV2(
            random_state=random_state, n_jobs=n_jobs, time_limit_in_minutes=fit_contract
        )

    # Interval based
    elif c == "rise-500":
        from sktime.classification.interval_based import RandomIntervalSpectralEnsemble

        return RandomIntervalSpectralEnsemble(
            n_estimators=500, random_state=random_state, n_jobs=n_jobs
        )
    elif (
        c == "rise"
        or c == "randomintervalspectralforest"
        or c == "randomintervalspectralensemble"
    ):
        from sktime.classification.interval_based import RandomIntervalSpectralEnsemble

        return RandomIntervalSpectralEnsemble(random_state=random_state, n_jobs=n_jobs)
    elif c == "tsf-500":
        from sktime.classification.interval_based import TimeSeriesForestClassifier

        return TimeSeriesForestClassifier(
            n_estimators=500, random_state=random_state, n_jobs=n_jobs
        )
    elif c == "tsf" or c == "timeseriesforestclassifier":
        from sktime.classification.interval_based import TimeSeriesForestClassifier

        return TimeSeriesForestClassifier(random_state=random_state, n_jobs=n_jobs)
    elif c == "cif-500":
        from sktime.classification.interval_based import CanonicalIntervalForest

        return CanonicalIntervalForest(
            n_estimators=500, random_state=random_state, n_jobs=n_jobs
        )
    elif c == "cif" or c == "canonicalintervalforest":
        from sktime.classification.interval_based import CanonicalIntervalForest

        return CanonicalIntervalForest(random_state=random_state, n_jobs=n_jobs)
    elif c == "stsf-500":
        from sktime.classification.interval_based import SupervisedTimeSeriesForest

        return SupervisedTimeSeriesForest(
            n_estimators=500, random_state=random_state, n_jobs=n_jobs
        )
    elif c == "stsf" or c == "supervisedtimeseriesforest":
        from sktime.classification.interval_based import SupervisedTimeSeriesForest

        return SupervisedTimeSeriesForest(random_state=random_state, n_jobs=n_jobs)
    elif c == "drcif-500":
        from sktime.classification.interval_based import DrCIF

        return DrCIF(
            n_estimators=500,
            random_state=random_state,
            save_transformed_data=build_train_file,
            n_jobs=n_jobs,
            time_limit_in_minutes=fit_contract,
        )
    elif c == "drcif":
        from sktime.classification.interval_based import DrCIF

        return DrCIF(
            random_state=random_state,
            save_transformed_data=build_train_file,
            n_jobs=n_jobs,
            time_limit_in_minutes=fit_contract,
        )

    # Shapelet based
    elif c == "stc-2hour":
        from sktime.classification.shapelet_based import ShapeletTransformClassifier

        return ShapeletTransformClassifier(
            transform_limit_in_minutes=120,
            random_state=random_state,
            save_transformed_data=build_train_file,
            n_jobs=n_jobs,
        )
    elif c == "stc" or c == "shapelettransformclassifier":
        from sktime.classification.shapelet_based import ShapeletTransformClassifier

        return ShapeletTransformClassifier(
            random_state=random_state,
            save_transformed_data=build_train_file,
            n_jobs=n_jobs,
            time_limit_in_minutes=fit_contract,
        )
    elif c == "rdst":
        from tsml_eval.sktime_estimators.classification.shapelet_based.rdst import RDST

        return RDST(random_state=random_state)
    elif c == "rdst-ensemble":
        from tsml_eval.sktime_estimators.classification.shapelet_based.rdst import (
            RDSTEnsemble,
        )

        return RDSTEnsemble(random_state=random_state)
    elif c == "rsf":
        from tsml_eval.sktime_estimators.classification.shapelet_based.rsf import (
            RandomShapeletForest,
        )

        return RandomShapeletForest(random_state=random_state)

    # Deep learning based
    elif c == "cnn" or c == "cnnclassifier":
        from sktime.classification.deep_learning.cnn import CNNClassifier

        return CNNClassifier(random_state=random_state)
    elif c == "fcnn" or c == "fcnclassifier":
        from sktime.classification.deep_learning.fcn import FCNClassifier

        return FCNClassifier(random_state=random_state)
    elif c == "mlp" or c == "mlpclassifier":
        from sktime.classification.deep_learning.mlp import MLPClassifier

        return MLPClassifier(random_state=random_state)
    elif c == "tapnet" or c == "tapnetclassifier":
        from sktime.classification.deep_learning.tapnet import TapNetClassifier

        return TapNetClassifier(random_state=random_state)
    # Other
    elif c == "dummy" or c == "dummyclassifier":
        from sktime.classification.dummy import DummyClassifier

        return DummyClassifier()
    elif c == "composabletimeseriesforestclassifier":
        from sktime.classification.compose import ComposableTimeSeriesForestClassifier

        return ComposableTimeSeriesForestClassifier(random_state=random_state)

    # requires constructor arguments
    elif c == "columnensemble" or c == "columnensembleclassifier":
        raise Exception(
            "Cannot create a ColumnEnsembleClassifier without passing a base "
            "classifier "
        )
    elif c == "probabilitythresholdearlyclassifier":
        raise Exception(
            "probabilitythresholdearlyclassifier is for early classification, "
            "not applicable here"
        )
    elif c == "classifierpipeline" or c == "sklearnclassifierpipeline":
        raise Exception(
            "Cannot create a ClassifierPipeline or SklearnClassifierPipeline "
            "without passing a base "
            "classifier and transform(s)"
        )
    elif c == "weightedensembleclassifier" or c == "weightedensemble":
        raise Exception(
            "Cannot create a WeightedEnsembleClassifier"
            "without passing base classifiers"
        )

    # Reading results from file
    elif c == "fromfile":
        from tsml_eval._wip.estimator_from_file.hivecote import FromFileHIVECOTE

        file_paths = [
            "tsml_eval/_wip.estimator_from_file/tests/test_files/Arsenal/",
            "tsml_eval/_wip.estimator_from_file/tests/test_files/DrCIF/",
            "tsml_eval/_wip.estimator_from_file/tests/test_files/STC/",
            "tsml_eval/_wip.estimator_from_file/tests/test_files/TDE/",
        ]

        return FromFileHIVECOTE(file_paths=file_paths, random_state=0)

    # invalid classifier
    else:
        raise Exception("UNKNOWN CLASSIFIER ", c, " in set_classifier")
