# -*- coding: utf-8 -*-
"""Set classifier function."""

__author__ = ["TonyBagnall", "MatthewMiddlehurst"]


def set_classifier(cls, resample_id=None, train_file=False, n_jobs=1, contract=0):
    """Construct a classifier, possibly seeded.

    Basic way of creating the classifier to build using the default settings. This
    set up is to help with batch jobs for multiple problems to facilitate easy
    reproducibility for use with load_and_run_classification_experiment. You can pass a
    classifier object instead to run_classification_experiment.
    TODO: add contract and checkpoint options

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
        from sktime.classification.dictionary_based import BOSSEnsemble

        return BOSSEnsemble(
            random_state=resample_id, n_jobs=n_jobs, save_train_predictions=train_file
        )
    elif name == "individualboss":
        from sktime.classification.dictionary_based import IndividualBOSS

        return IndividualBOSS(
            random_state=resample_id,
            n_jobs=n_jobs,
        )
    elif name == "cboss" or name == "contractableboss":
        from sktime.classification.dictionary_based import ContractableBOSS

        return ContractableBOSS(random_state=resample_id, n_jobs=n_jobs)
    elif name == "tde" or name == "temporaldictionaryensemble":
        from sktime.classification.dictionary_based import TemporalDictionaryEnsemble

        return TemporalDictionaryEnsemble(
            random_state=resample_id,
            save_train_predictions=train_file,
            n_jobs=n_jobs,
            time_limit_in_minutes=contract,
        )
    elif name == "individualtde":
        from sktime.classification.dictionary_based import IndividualTDE

        return IndividualTDE(random_state=resample_id, n_jobs=n_jobs)
    elif name == "weasel":
        from sktime.classification.dictionary_based import WEASEL

        return WEASEL(random_state=resample_id, n_jobs=n_jobs)
    elif name == "weasel-logistic":
        from sktime.classification.dictionary_based import WEASEL

        return WEASEL(
            random_state=resample_id, n_jobs=n_jobs, support_probabilities=True
        )
    elif name == "muse":
        from sktime.classification.dictionary_based import MUSE

        return MUSE(random_state=resample_id, n_jobs=n_jobs)
    elif name == "muse-logistic":
        from sktime.classification.dictionary_based import MUSE

        return MUSE(random_state=resample_id, n_jobs=n_jobs, support_probabilities=True)
    # Distance based
    elif name == "pf" or name == "proximityforest":
        from sktime.classification.distance_based import ProximityForest

        return ProximityForest(random_state=resample_id)
    elif name == "pt" or name == "proximitytree":
        from sktime.classification.distance_based import ProximityTree

        return ProximityTree(random_state=resample_id)
    elif name == "ps" or name == "proximitystump":
        from sktime.classification.distance_based import ProximityStump

        return ProximityStump(random_state=resample_id)
    elif name == "dtw" or name == "1nn-dtw" or name == "kneighborstimeseriesclassifier":
        from sktime.classification.distance_based import KNeighborsTimeSeriesClassifier

        return KNeighborsTimeSeriesClassifier()
    elif name == "ed" or name == "1nn-ed":
        from sktime.classification.distance_based import KNeighborsTimeSeriesClassifier

        return KNeighborsTimeSeriesClassifier(distance="ed")
    elif name == "msm" or name == "1nn-msm":
        from sktime.classification.distance_based import KNeighborsTimeSeriesClassifier

        return KNeighborsTimeSeriesClassifier(distance="msm")
    elif name == "ee" or name == "elasticensemble":
        from sktime.classification.distance_based import ElasticEnsemble

        return ElasticEnsemble(random_state=resample_id)
    elif name == "shapedtw":
        from sktime.classification.distance_based import ShapeDTW

        return ShapeDTW()
    # Feature based
    elif name == "summary-500":
        from sklearn.ensemble import RandomForestClassifier
        from sktime.classification.feature_based import SummaryClassifier

        return SummaryClassifier(
            random_state=resample_id, estimator=RandomForestClassifier(n_estimators=500)
        )
    elif name == "summaryclassifier" or name == "summary":
        from sktime.classification.feature_based import SummaryClassifier

        return SummaryClassifier(random_state=resample_id)
    elif name == "summary-intervals":
        from sklearn.ensemble import RandomForestClassifier
        from sktime.classification.feature_based import RandomIntervalClassifier
        from sktime.transformations.series.summarize import SummaryTransformer

        return RandomIntervalClassifier(
            random_state=resample_id,
            interval_transformers=SummaryTransformer(
                summary_function=("mean", "std", "min", "max"),
                quantiles=(0.25, 0.5, 0.75),
            ),
            estimator=RandomForestClassifier(n_estimators=500),
        )
    elif name == "randominterval-rf" or name == "catch22intervals-rf":
        from sklearn.ensemble import RandomForestClassifier
        from sktime.classification.feature_based import RandomIntervalClassifier

        return RandomIntervalClassifier(
            random_state=resample_id, estimator=RandomForestClassifier(n_estimators=500)
        )
    elif (
        name == "randomintervalclassifier"
        or name == "randominterval"
        or name == "catch22intervals"
    ):
        from sktime.classification.feature_based import RandomIntervalClassifier

        return RandomIntervalClassifier(random_state=resample_id)
    elif name == "catch22-500":
        from sklearn.ensemble import RandomForestClassifier
        from sktime.classification.feature_based import Catch22Classifier

        return Catch22Classifier(
            random_state=resample_id, estimator=RandomForestClassifier(n_estimators=500)
        )
    elif name == "catch22" or name == "catch22classifier":
        from sktime.classification.feature_based import Catch22Classifier

        return Catch22Classifier(random_state=resample_id)
    elif name == "matrixprofile" or name == "matrixprofileclassifier":
        from sktime.classification.feature_based import MatrixProfileClassifier

        return MatrixProfileClassifier(random_state=resample_id)
    elif name == "freshprince":
        from sktime.classification.feature_based import FreshPRINCE

        return FreshPRINCE(random_state=resample_id)
    elif name == "tsfresh-nofs":
        from sktime.classification.feature_based import TSFreshClassifier

        return TSFreshClassifier(
            random_state=resample_id, relevant_feature_extractor=False
        )
    elif name == "tsfreshclassifier" or name == "tsfresh":
        from sktime.classification.feature_based import TSFreshClassifier

        return TSFreshClassifier(random_state=resample_id)
    elif name == "signatureclassifier" or name == "signatures":
        from sktime.classification.feature_based import SignatureClassifier

        return SignatureClassifier(random_state=resample_id)
    # hybrids
    elif name == "hc1" or name == "hivecotev1":
        from sktime.classification.hybrid import HIVECOTEV1

        return HIVECOTEV1(random_state=resample_id)
    elif name == "hc2" or name == "hivecotev2":
        from sktime.classification.hybrid import HIVECOTEV2

        return HIVECOTEV2(random_state=resample_id)
    # Interval based
    elif name == "rise-500":
        from sktime.classification.interval_based import RandomIntervalSpectralEnsemble

        return RandomIntervalSpectralEnsemble(
            random_state=resample_id, n_estimators=500, n_jobs=n_jobs
        )
    elif (
        name == "rise"
        or name == "randomintervalspectralforest"
        or name == "randomintervalspectralensemble"
    ):
        from sktime.classification.interval_based import RandomIntervalSpectralEnsemble

        return RandomIntervalSpectralEnsemble(random_state=resample_id, n_jobs=n_jobs)
    elif name == "tsf-500":
        from sktime.classification.interval_based import TimeSeriesForestClassifier

        return TimeSeriesForestClassifier(
            random_state=resample_id, n_estimators=500, n_jobs=n_jobs
        )
    elif name == "tsf" or name == "timeseriesforestclassifier":
        from sktime.classification.interval_based import TimeSeriesForestClassifier

        return TimeSeriesForestClassifier(random_state=resample_id, n_jobs=n_jobs)
    elif name == "cif-500":
        from sktime.classification.interval_based import CanonicalIntervalForest

        return CanonicalIntervalForest(
            random_state=resample_id, n_estimators=500, n_jobs=n_jobs
        )
    elif name == "cif" or name == "canonicalintervalforest":
        from sktime.classification.interval_based import CanonicalIntervalForest

        return CanonicalIntervalForest(random_state=resample_id, n_jobs=n_jobs)
    elif name == "stsf-500":
        from sktime.classification.interval_based import SupervisedTimeSeriesForest

        return SupervisedTimeSeriesForest(
            random_state=resample_id, n_estimators=500, n_jobs=n_jobs
        )
    elif name == "stsf" or name == "supervisedtimeseriesforest":
        from sktime.classification.interval_based import SupervisedTimeSeriesForest

        return SupervisedTimeSeriesForest(random_state=resample_id, n_jobs=n_jobs)
    elif name == "drcif-500":
        from sktime.classification.interval_based import DrCIF

        return DrCIF(
            random_state=resample_id,
            n_estimators=500,
            save_transformed_data=train_file,
            n_jobs=n_jobs,
        )
    elif name == "drcif":
        from sktime.classification.interval_based import DrCIF

        return DrCIF(
            random_state=resample_id, save_transformed_data=train_file, n_jobs=n_jobs
        )
    # Convolution based
    elif name == "rocket" or name == "rocketclassifier":
        from sktime.classification.kernel_based import RocketClassifier

        return RocketClassifier(random_state=resample_id)
    elif name == "mini-rocket":
        from sktime.classification.kernel_based import RocketClassifier

        return RocketClassifier(random_state=resample_id, rocket_transform="minirocket")
    elif name == "multi-rocket":
        from sktime.classification.kernel_based import RocketClassifier

        return RocketClassifier(
            random_state=resample_id, rocket_transform="multirocket"
        )
    elif name == "arsenal":
        from sktime.classification.kernel_based import Arsenal

        return Arsenal(random_state=resample_id, save_transformed_data=train_file)
    elif name == "mini-arsenal":
        from sktime.classification.kernel_based import Arsenal

        return Arsenal(
            random_state=resample_id,
            save_transformed_data=train_file,
            rocket_transform="minirocket",
        )
    elif name == "multi-arsenal":
        from sktime.classification.kernel_based import Arsenal

        return Arsenal(
            random_state=resample_id,
            save_transformed_data=train_file,
            rocket_transform="multirocket",
        )
    # Shapelet based
    elif name == "stc-2hour":
        from sktime.classification.shapelet_based import ShapeletTransformClassifier

        return ShapeletTransformClassifier(
            transform_limit_in_minutes=120,
            random_state=resample_id,
            save_transformed_data=train_file,
        )
    elif name == "stc" or name == "shapelettransformclassifier":
        from sktime.classification.shapelet_based import ShapeletTransformClassifier

        return ShapeletTransformClassifier(
            random_state=resample_id,
            save_transformed_data=train_file,
        )
    # Deep learning based
    elif name == "cnn" or name == "cnnclassifier":
        from sktime.classification.deep_learning.cnn import CNNClassifier

        return CNNClassifier(random_state=resample_id)
    elif name == "fcnn" or name == "fcnclassifier":
        from sktime.classification.deep_learning.fcn import FCNClassifier

        return FCNClassifier(random_state=resample_id)
    elif name == "mlp" or name == "mlpclassifier":
        from sktime.classification.deep_learning.mlp import MLPClassifier

        return MLPClassifier(random_state=resample_id)
    elif name == "tapnet" or name == "tapnetclassifier":
        from sktime.classification.deep_learning.tapnet import TapNetClassifier

        return TapNetClassifier(random_state=resample_id)
    # Other
    elif name == "dummy" or name == "dummyclassifier":
        from sktime.classification.dummy import DummyClassifier

        return DummyClassifier()
    elif name == "composabletimeseriesforestclassifier":
        from sktime.classification.compose import ComposableTimeSeriesForestClassifier

        return ComposableTimeSeriesForestClassifier(random_state=resample_id)

    # requires constructor arguments
    elif name == "columnensemble" or name == "columnensembleclassifier":
        raise Exception(
            "Cannot create a ColumnEnsembleClassifier without passing a base "
            "classifier "
        )
    elif name == "probabilitythresholdearlyclassifier":
        raise Exception(
            "probabilitythresholdearlyclassifier is for early classification, "
            "not applicable here"
        )
    elif name == "classifierpipeline" or name == "sklearnclassifierpipeline":
        raise Exception(
            "Cannot create a ClassifierPipeline or SklearnClassifierPipeline "
            "without passing a base "
            "classifier and transform(s)"
        )
    elif name == "weightedensembleclassifier" or name == "weightedensemble":
        raise Exception(
            "Cannot create a WeightedEnsembleClassifier"
            "without passing base classifiers"
        )

    # Non-sktime package estimators
    elif name == "weasel-dilation":
        from tsml_estimator_evaluation.sktime_estimators.classification.weasel_dilation import (  # noqa: E501
            WEASEL_DILATION,
        )

        return WEASEL_DILATION()
    elif name == "muse-dilation":
        from tsml_estimator_evaluation.sktime_estimators.classification.muse_dilation import (  # noqa: E501
            MUSE_DILATION,
        )

        return MUSE_DILATION()
    elif name == "rdst":
        from tsml_estimator_evaluation.sktime_estimators.classification.rdst import RDST

        return RDST()
    elif name == "rdst-ensemble":
        from tsml_estimator_evaluation.sktime_estimators.classification.rdst import (
            RDSTEnsemble,
        )

        return RDSTEnsemble()
    elif name == "hydra":
        from tsml_estimator_evaluation.sktime_estimators.classification.hydra import (
            HYDRA,
        )

        return HYDRA()

    else:
        raise Exception("UNKNOWN CLASSIFIER ", name, " in set_classifier")
