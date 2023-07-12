"""Classifiers used in the publication."""

__author__ = ["TonyBagnall", "MatthewMiddlehurst"]

from tsml_eval.utils.functions import str_in_nested_list

bakeoff_classifiers = [
    # distance based
    ["KNeighborsTimeSeriesClassifier", "dtw", "1nn-dtw"],
    "ShapeDTW",
    # feature based
    ["Catch22Classifier", "catch22"],
    ["FreshPRINCEClassifier", "freshprince"],
    ["TSFreshClassifier", "tsfresh"],
    ["SignatureClassifier", "signatures"],
    # shapelet based
    ["ShapeletTransformClassifier", "stc", "stc-2hour"],
    "RDST",
    ["RandomShapeletForestClassifier", "randomshapeletforest", "rsf"],
    ["MrSQMClassifier", "mrsqm"],
    # interval based
    ["RSTSFClassifier", "rstsf", "r-stsf"],
    ["RandomIntervalSpectralEnsemble", "rise"],
    ["TimeSeriesForestClassifier", "tsf"],
    ["CanonicalIntervalForest", "cif"],
    ["SupervisedTimeSeriesForest", "stsf"],
    "DrCIF",
    # dictionary based
    ["BOSSEnsemble", "boss"],
    ["ContractableBOSS", "cboss"],
    ["TemporalDictionaryEnsemble", "tde"],
    "WEASEL",
    ["WEASEL_V2", "weaseldilation", "weasel-dilation", "weasel-d"],
    # convolution based
    ["RocketClassifier", "rocket"],
    ["minirocket", "mini-rocket"],
    ["multirocket", "multi-rocket"],
    ["arsenalclassifier", "Arsenal"],
    "HYDRA",
    ["HydraMultiRocket", "hydra-multirocket"],
    # deep learning
    ["CNNClassifier", "cnn"],
    ["ResNetClassifier", "resnet"],
    ["InceptionTimeClassifier", "inceptiontime"],
    # hybrid
    ["HIVECOTEV1", "hc1"],
    ["HIVECOTEV2", "hc2"],
]


def _set_bakeoff_classifier(
    classifier_name,
    random_state=None,
    n_jobs=1,
):
    c = classifier_name.lower()

    if not str_in_nested_list(bakeoff_classifiers, c):
        raise Exception("UNKNOWN CLASSIFIER ", c, " in set_bakeoff_classifier")

    if c == "kneighborstimeseriesclassifier" or c == "dtw" or c == "1nn-dtw":
        from aeon.classification.distance_based import KNeighborsTimeSeriesClassifier

        return KNeighborsTimeSeriesClassifier(distance="dtw", n_jobs=n_jobs)
    elif c == "shapedtw":
        from aeon.classification.distance_based import ShapeDTW

        return ShapeDTW()
    elif c == "catch22classifier" or c == "catch22":
        from aeon.classification.feature_based import Catch22Classifier
        from sklearn.ensemble import RandomForestClassifier

        return Catch22Classifier(
            estimator=RandomForestClassifier(n_estimators=500),
            random_state=random_state,
            n_jobs=n_jobs,
        )
    elif c == "freshprinceclassifier" or c == "freshprince":
        from tsml_eval.estimators.classification.feature_based import (
            FreshPRINCEClassifier,
        )

        return FreshPRINCEClassifier(random_state=random_state, n_jobs=n_jobs)
    elif c == "tsfreshclassifier" or c == "tsfresh":
        from aeon.classification.feature_based import TSFreshClassifier

        return TSFreshClassifier(random_state=random_state, n_jobs=n_jobs)
    elif c == "signatureclassifier" or c == "signatures":
        from aeon.classification.feature_based import SignatureClassifier

        return SignatureClassifier(random_state=random_state)
    elif c == "shapelettransformclassifier" or c == "stc" or c == "stc-2hour":
        from tsml_eval.estimators.classification.shapelet_based import (
            ShapeletTransformClassifier,
        )

        return ShapeletTransformClassifier(
            transform_limit_in_minutes=120,
            random_state=random_state,
            n_jobs=n_jobs,
        )
    elif c == "rdst":
        from tsml_eval.estimators.classification.shapelet_based.rdst import RDST

        return RDST(random_state=random_state)
    elif (
        c == "randomshapeletforestclassifier"
        or c == "randomshapeletforest"
        or c == "rsf"
    ):
        from tsml.shapelet_based import RandomShapeletForestClassifier

        return RandomShapeletForestClassifier(random_state=random_state, n_jobs=n_jobs)
    elif c == "mrsqmclassifier" or c == "mrsqm":
        from aeon.classification.shapelet_based import MrSQMClassifier

        return MrSQMClassifier(random_state=random_state)
    elif c == c == "rstsfclassifier" or c == "rstsf" or c == "r-stsf":
        from tsml.interval_based import RSTSFClassifier

        return RSTSFClassifier(
            n_estimators=500, random_state=random_state, n_jobs=n_jobs
        )
    elif c == "randomintervalspectralensemble" or c == "rise":
        from aeon.classification.interval_based import RandomIntervalSpectralEnsemble

        return RandomIntervalSpectralEnsemble(
            n_estimators=500, random_state=random_state, n_jobs=n_jobs
        )
    elif c == "timeseriesforestclassifier" or c == "tsf":
        from aeon.classification.interval_based import TimeSeriesForestClassifier

        return TimeSeriesForestClassifier(
            n_estimators=500, random_state=random_state, n_jobs=n_jobs
        )
    elif c == "canonicalintervalforest" or c == "cif":
        from aeon.classification.interval_based import CanonicalIntervalForest

        return CanonicalIntervalForest(random_state=random_state, n_jobs=n_jobs)
    elif c == "supervisedtimeseriesforest" or c == "stsf":
        from aeon.classification.interval_based import SupervisedTimeSeriesForest

        return SupervisedTimeSeriesForest(
            n_estimators=500, random_state=random_state, n_jobs=n_jobs
        )
    elif c == "drcif":
        from aeon.classification.interval_based import DrCIF

        return DrCIF(
            n_estimators=500,
            random_state=random_state,
            n_jobs=n_jobs,
        )
    elif c == "bossensemble" or c == "boss":
        from aeon.classification.dictionary_based import BOSSEnsemble

        return BOSSEnsemble(
            random_state=random_state,
            n_jobs=n_jobs,
        )
    elif c == "contractableboss" or c == "cboss":
        from aeon.classification.dictionary_based import ContractableBOSS

        return ContractableBOSS(
            random_state=random_state,
            n_jobs=n_jobs,
        )
    elif c == "temporaldictionaryensemble" or c == "tde":
        from aeon.classification.dictionary_based import TemporalDictionaryEnsemble

        return TemporalDictionaryEnsemble(
            random_state=random_state,
            n_jobs=n_jobs,
        )
    elif c == "weasel":
        from aeon.classification.dictionary_based import WEASEL

        return WEASEL(
            random_state=random_state,
            n_jobs=n_jobs,
            support_probabilities=True,
        )
    elif (
        c == "weasel_v2"
        or c == "weaseldilation"
        or c == "weasel-dilation"
        or c == "weasel-d"
    ):
        from aeon.classification.dictionary_based import WEASEL_V2

        return WEASEL_V2(random_state=random_state, n_jobs=n_jobs)
    elif c == "rocketclassifier" or c == "rocket":
        from aeon.classification.convolution_based import RocketClassifier

        return RocketClassifier(random_state=random_state, n_jobs=n_jobs)
    elif c == "minirocket" or c == "mini-rocket":
        from aeon.classification.convolution_based import RocketClassifier

        return RocketClassifier(
            rocket_transform="minirocket",
            random_state=random_state,
            n_jobs=n_jobs,
        )
    elif c == "multirocket" or c == "multi-rocket":
        from aeon.classification.convolution_based import RocketClassifier

        return RocketClassifier(
            rocket_transform="multirocket",
            random_state=random_state,
            n_jobs=n_jobs,
        )
    elif c == "arsenalclassifier" or c == "arsenal":
        from aeon.classification.convolution_based import Arsenal

        return Arsenal(
            random_state=random_state,
            n_jobs=n_jobs,
        )
    elif c == "hydra":
        from tsml_eval.estimators.classification.convolution_based.hydra import HYDRA

        return HYDRA(random_state=random_state, n_jobs=n_jobs)
    elif c == "hydramultirocket" or c == "hydra-multirocket":
        from tsml_eval.estimators.classification.convolution_based.hydra import (
            HydraMultiRocket,
        )

        return HydraMultiRocket(random_state=random_state, n_jobs=n_jobs)
    elif c == "cnnclassifier" or c == "cnn":
        from tsml_eval.estimators.classification.deep_learning import CNNClassifier

        return CNNClassifier(random_state=random_state)
    elif c == "resnetclassifier" or c == "resnet":
        from tsml_eval.estimators.classification.deep_learning.resnet import (
            ResNetClassifier,
        )

        return ResNetClassifier(random_state=random_state)
    elif c == "inceptiontimeclassifier" or c == "inceptiontime":
        from tsml_eval.estimators.classification.deep_learning.inception_time import (
            InceptionTimeClassifier,
        )

        return InceptionTimeClassifier(random_state=random_state)
    elif c == "hivecotev1" or c == "hc1":
        from tsml_eval.estimators.classification.hybrid import HIVECOTEV1

        return HIVECOTEV1(random_state=random_state, n_jobs=n_jobs)
    elif c == "hivecotev2" or c == "hc2":
        from aeon.classification.hybrid import HIVECOTEV2

        return HIVECOTEV2(
            random_state=random_state,
            n_jobs=n_jobs,
        )
