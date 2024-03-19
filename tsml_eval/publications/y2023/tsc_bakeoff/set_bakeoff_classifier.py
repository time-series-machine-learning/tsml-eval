"""Classifiers used in the publication."""

__author__ = ["TonyBagnall", "MatthewMiddlehurst"]

from tsml_eval.utils.functions import str_in_nested_list

bakeoff_classifiers = [
    # distance based
    ["KNeighborsTimeSeriesClassifier", "dtw", "1nn-dtw"],
    "ShapeDTW",
    ["GRAILClassifier", "grail"],
    # feature based
    ["Catch22Classifier", "catch22"],
    ["FreshPRINCEClassifier", "freshprince"],
    ["TSFreshClassifier", "tsfresh"],
    ["SignatureClassifier", "signatures"],
    # shapelet based
    ["ShapeletTransformClassifier", "stc", "stc-2hour"],
    ["RDSTClassifier", "rdst"],
    ["RandomShapeletForestClassifier", "randomshapeletforest", "rsf"],
    ["MrSQMClassifier", "mrsqm"],
    # interval based
    ["RSTSFClassifier", "rstsf", "r-stsf"],
    ["RandomIntervalSpectralEnsembleClassifier", "rise"],
    ["TimeSeriesForestClassifier", "tsf"],
    ["CanonicalIntervalForestClassifier", "cif"],
    ["SupervisedTimeSeriesForest", "stsf"],
    ["drcif", "DrCIFClassifier"],
    ["quant", "QUANTClassifier"],
    # dictionary based
    ["BOSSEnsemble", "boss"],
    ["ContractableBOSS", "cboss"],
    ["TemporalDictionaryEnsemble", "tde"],
    ["WEASEL", "weasel v1", "weasel v1.0"],
    ["weasel_v2", "weasel v2", "weasel v2.0", "weasel-dilation", "weasel-d"],
    # convolution based
    ["RocketClassifier", "rocket"],
    ["minirocket", "mini-rocket"],
    ["multirocket", "multi-rocket"],
    ["arsenalclassifier", "Arsenal"],
    "HYDRA",
    ["mr-hydra", "multirockethydra", "multirocket-hydra"],
    # deep learning
    ["CNNClassifier", "cnn"],
    ["ResNetClassifier", "resnet"],
    ["InceptionTimeClassifier", "inceptiontime"],
    ["h-inceptiontimeclassifier", "h-inceptiontime"],
    ["LITETimeClassifier", "litetime"],
    # hybrid
    ["HIVECOTEV1", "hc1"],
    ["HIVECOTEV2", "hc2"],
    ["RIST", "RISTClassifier"],
]


def _set_bakeoff_classifier(
    classifier_name,
    random_state=None,
    n_jobs=1,
    **kwargs,
):
    c = classifier_name.lower()

    if not str_in_nested_list(bakeoff_classifiers, c):
        raise ValueError(f"UNKNOWN CLASSIFIER: {c} in _set_bakeoff_classifier")

    if c == "kneighborstimeseriesclassifier" or c == "dtw" or c == "1nn-dtw":
        from aeon.classification.distance_based import KNeighborsTimeSeriesClassifier

        return KNeighborsTimeSeriesClassifier(
            distance="dtw", n_neighbors=1, n_jobs=n_jobs, **kwargs
        )
    elif c == "shapedtw":
        from aeon.classification.distance_based import ShapeDTW

        return ShapeDTW(
            **kwargs,
        )
    elif c == "grailclassifier" or c == "grail":
        from tsml.distance_based import GRAILClassifier

        return GRAILClassifier(**kwargs)
    elif c == "catch22classifier" or c == "catch22":
        from aeon.classification.feature_based import Catch22Classifier
        from sklearn.ensemble import RandomForestClassifier

        return Catch22Classifier(
            estimator=RandomForestClassifier(n_estimators=500),
            catch24=False,
            random_state=random_state,
            n_jobs=n_jobs,
            **kwargs,
        )
    elif c == "freshprinceclassifier" or c == "freshprince":
        from aeon.classification.feature_based import FreshPRINCEClassifier

        return FreshPRINCEClassifier(random_state=random_state, n_jobs=n_jobs, **kwargs)
    elif c == "tsfreshclassifier" or c == "tsfresh":
        from aeon.classification.feature_based import TSFreshClassifier

        return TSFreshClassifier(random_state=random_state, n_jobs=n_jobs, **kwargs)
    elif c == "signatureclassifier" or c == "signatures":
        from aeon.classification.feature_based import SignatureClassifier

        return SignatureClassifier(
            random_state=random_state,
            **kwargs,
        )
    elif c == "shapelettransformclassifier" or c == "stc" or c == "stc-2hour":
        from aeon.classification.shapelet_based import ShapeletTransformClassifier

        return ShapeletTransformClassifier(
            n_shapelet_samples=10000,
            random_state=random_state,
            n_jobs=n_jobs,
            **kwargs,
        )
    elif c == "rdstclassifier" or c == "rdst":
        from aeon.classification.shapelet_based import RDSTClassifier

        return RDSTClassifier(
            random_state=random_state,
            **kwargs,
        )
    elif (
        c == "randomshapeletforestclassifier"
        or c == "randomshapeletforest"
        or c == "rsf"
    ):
        from tsml.shapelet_based import RandomShapeletForestClassifier

        return RandomShapeletForestClassifier(
            random_state=random_state,
            n_jobs=n_jobs,
            **kwargs,
        )
    elif c == "mrsqmclassifier" or c == "mrsqm":
        from aeon.classification.shapelet_based import MrSQMClassifier

        return MrSQMClassifier(
            random_state=random_state,
            **kwargs,
        )
    elif c == c == "rstsfclassifier" or c == "rstsf" or c == "r-stsf":
        from tsml.interval_based import RSTSFClassifier

        return RSTSFClassifier(
            n_estimators=500,
            random_state=random_state,
            n_jobs=n_jobs,
            **kwargs,
        )
    elif c == "randomintervalspectralensembleclassifier" or c == "rise":
        from aeon.classification.interval_based import (
            RandomIntervalSpectralEnsembleClassifier,
        )

        return RandomIntervalSpectralEnsembleClassifier(
            n_estimators=500,
            min_interval_length=16,
            random_state=random_state,
            n_jobs=n_jobs,
            **kwargs,
        )
    elif c == "timeseriesforestclassifier" or c == "tsf":
        from aeon.classification.interval_based import TimeSeriesForestClassifier

        return TimeSeriesForestClassifier(
            n_estimators=500,
            random_state=random_state,
            n_jobs=n_jobs,
            **kwargs,
        )
    elif c == "canonicalintervalforestclassifier" or c == "cif":
        from aeon.classification.interval_based import CanonicalIntervalForestClassifier

        return CanonicalIntervalForestClassifier(
            random_state=random_state,
            n_jobs=n_jobs,
            **kwargs,
        )
    elif c == "supervisedtimeseriesforest" or c == "stsf":
        from aeon.classification.interval_based import SupervisedTimeSeriesForest

        return SupervisedTimeSeriesForest(
            n_estimators=500,
            random_state=random_state,
            n_jobs=n_jobs,
            **kwargs,
        )
    elif c == "drcif" or c == "drcifclassifier":
        from aeon.classification.interval_based import DrCIFClassifier

        return DrCIFClassifier(
            n_estimators=500,
            random_state=random_state,
            n_jobs=n_jobs,
            **kwargs,
        )
    elif c == "quantclassifier" or c == "quant":
        from tsml_eval.estimators.classification.interval_based.quant import (
            QuantClassifier,
        )

        return QuantClassifier(random_state=random_state, **kwargs)
    elif c == "bossensemble" or c == "boss":
        from aeon.classification.dictionary_based import BOSSEnsemble

        return BOSSEnsemble(
            random_state=random_state,
            n_jobs=n_jobs,
            **kwargs,
        )
    elif c == "contractableboss" or c == "cboss":
        from aeon.classification.dictionary_based import ContractableBOSS

        return ContractableBOSS(
            random_state=random_state,
            n_jobs=n_jobs,
            **kwargs,
        )
    elif c == "temporaldictionaryensemble" or c == "tde":
        from aeon.classification.dictionary_based import TemporalDictionaryEnsemble

        return TemporalDictionaryEnsemble(
            random_state=random_state,
            n_jobs=n_jobs,
            **kwargs,
        )
    elif c == "weasel" or c == "weasel v1" or c == "weasel v1.0":
        from aeon.classification.dictionary_based import WEASEL

        return WEASEL(
            random_state=random_state,
            n_jobs=n_jobs,
            support_probabilities=True,
            **kwargs,
        )
    elif (
        c == "weasel_v2"
        or c == "weasel v2"
        or c == "weasel v2.0"
        or c == "weasel-dilation"
        or c == "weasel-d"
    ):
        from aeon.classification.dictionary_based import WEASEL_V2

        return WEASEL_V2(
            random_state=random_state,
            n_jobs=n_jobs,
            **kwargs,
        )
    elif c == "rocketclassifier" or c == "rocket":
        from aeon.classification.convolution_based import RocketClassifier

        return RocketClassifier(
            random_state=random_state,
            n_jobs=n_jobs,
            **kwargs,
        )
    elif c == "minirocket" or c == "mini-rocket":
        from aeon.classification.convolution_based import RocketClassifier

        return RocketClassifier(
            rocket_transform="minirocket",
            random_state=random_state,
            n_jobs=n_jobs,
            **kwargs,
        )
    elif c == "multirocket" or c == "multi-rocket":
        from aeon.classification.convolution_based import RocketClassifier

        return RocketClassifier(
            rocket_transform="multirocket",
            random_state=random_state,
            n_jobs=n_jobs,
            **kwargs,
        )
    elif c == "arsenalclassifier" or c == "arsenal":
        from aeon.classification.convolution_based import Arsenal

        return Arsenal(
            random_state=random_state,
            n_jobs=n_jobs,
            **kwargs,
        )
    elif c == "hydra":
        from tsml_eval.estimators.classification.convolution_based.hydra import HYDRA

        return HYDRA(
            random_state=random_state,
            n_jobs=n_jobs,
            **kwargs,
        )
    elif c == "mr-hydra" or c == "multirockethydra" or c == "multirocket-hydra":
        from tsml_eval.estimators.classification.convolution_based.hydra import (
            MultiRocketHydra,
        )

        return MultiRocketHydra(
            random_state=random_state,
            n_jobs=n_jobs,
            **kwargs,
        )
    elif c == "cnnclassifier" or c == "cnn":
        from aeon.classification.deep_learning import CNNClassifier

        return CNNClassifier(
            random_state=random_state,
            **kwargs,
        )
    elif c == "resnetclassifier" or c == "resnet":
        from aeon.classification.deep_learning import ResNetClassifier

        return ResNetClassifier(
            random_state=random_state,
            **kwargs,
        )
    elif c == "inceptiontimeclassifier" or c == "inceptiontime":
        from aeon.classification.deep_learning import InceptionTimeClassifier

        return InceptionTimeClassifier(
            random_state=random_state,
            **kwargs,
        )
    elif c == "h-inceptiontimeclassifier" or c == "h-inceptiontime":
        from aeon.classification.deep_learning import InceptionTimeClassifier

        return InceptionTimeClassifier(
            use_custom_filters=True, random_state=random_state, **kwargs
        )
    elif c == "litetimeclassifier" or c == "litetime":
        from aeon.classification.deep_learning import LITETimeClassifier

        return LITETimeClassifier(random_state=random_state, **kwargs)
    elif c == "hivecotev1" or c == "hc1":
        from aeon.classification.hybrid import HIVECOTEV1

        return HIVECOTEV1(
            random_state=random_state,
            n_jobs=n_jobs,
            **kwargs,
        )
    elif c == "hivecotev2" or c == "hc2":
        from aeon.classification.hybrid import HIVECOTEV2

        return HIVECOTEV2(
            random_state=random_state,
            n_jobs=n_jobs,
            **kwargs,
        )
    elif c == "ristclassifier" or c == "rist":
        from sklearn.ensemble import ExtraTreesClassifier
        from tsml.hybrid import RISTClassifier

        return RISTClassifier(
            random_state=random_state,
            n_jobs=n_jobs,
            estimator=ExtraTreesClassifier(n_estimators=500, criterion="entropy"),
            **kwargs,
        )
