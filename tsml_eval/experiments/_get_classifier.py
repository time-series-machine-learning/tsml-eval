"""Get classifier function."""

__maintainer__ = ["TonyBagnall", "MatthewMiddlehurst"]

from tsml_eval.utils.functions import str_in_nested_list

convolution_based_classifiers = [
    ["rocketclassifier", "rocket"],
    ["minirocket", "mini-rocket", "minirocketclassifier"],
    ["multirocket", "multi-rocket", "multirocketclassifier"],
    ["arsenalclassifier", "arsenal"],
    ["miniarsenal", "mini-arsenal"],
    ["multiarsenal", "multi-arsenal"],
    ["hydraclassifier", "hydra"],
    ["multirockethydraclassifier", "multirockethydra", "multirocket-hydra", "mrhydra"],
]
deep_learning_classifiers = [
    ["timecnnclassifier", "timecnn", "cnnclassifier", "cnn"],
    ["fcnclassifier", "fcnn"],
    ["mlpclassifier", "mlp"],
    ["encoderclassifier", "encoder"],
    ["resnetclassifier", "resnet"],
    ["individualinceptionclassifier", "singleinception"],
    ["inceptiontimeclassifier", "inceptiontime"],
    ["h-inceptiontimeclassifier", "h-inceptiontime"],
    ["litetimeclassifier", "litetime"],
    "litetime-mv",
    ["individualliteclassifier", "individuallite"],
    ["disjointcnnclassifier", "disjointcnn"],
]
dictionary_based_classifiers = [
    ["bossensemble", "boss"],
    "individualboss",
    ["contractableboss", "cboss"],
    ["temporaldictionaryensemble", "tde"],
    "individualtde",
    "weasel",
    "weasel-logistic",
    "muse",
    "muse-logistic",
    ["weasel_v2", "weaseldilation", "weasel-dilation", "weasel-d"],
    "redcomets",
    "redcomets-500",
    ["mrseqlclassifier", "mrseql"],
    ["mrsqmclassifier", "mrsqm"],
]
distance_based_classifiers = [
    ["kneighborstimeseriesclassifier", "dtw", "1nn-dtw"],
    ["ed", "1nn-euclidean", "1nn-ed"],
    ["msm", "1nn-msm"],
    ["twe", "1nn-twe"],
    "1nn-dtw-cv",
    ["elasticensemble", "ee"],
    ["grailclassifier", "grail"],
    ["proximitytree", "proximitytreeclassifier"],
    ["proximityforest", "pf"],
]
feature_based_classifiers = [
    "summary-500",
    ["summaryclassifier", "summary"],
    "catch22-500",
    ["catch22classifier", "catch22"],
    "catch22-outlier",
    ["freshprinceclassifier", "freshprince"],
    "freshprince-500",
    "tsfresh-nofs",
    ["tsfreshclassifier", "tsfresh"],
    ["signatureclassifier", "signatures"],
]
hybrid_classifiers = [
    ["hivecotev1", "hc1"],
    ["hivecotev2", "hc2"],
    ["ristclassifier", "rist", "rist-extrat"],
]
interval_based_classifiers = [
    "rstsf-500",
    ["rstsfclassifier", "rstsf", "r-stsf"],
    "rise-500",
    ["randomintervalspectralensembleclassifier", "rise"],
    "tsf-500",
    ["timeseriesforestclassifier", "tsf"],
    "cif-500",
    ["canonicalintervalforestclassifier", "cif"],
    "stsf-500",
    ["supervisedtimeseriesforest", "stsf"],
    "drcif-500",
    ["drcif", "drcifclassifier"],
    "summary-intervals",
    ["randomintervals-500", "catch22-intervals-500"],
    ["randomintervalclassifier", "randomintervals", "catch22-intervals"],
    ["supervisedintervalclassifier", "supervisedintervals"],
    ["quantclassifier", "quant"],
]
other_classifiers = [
    ["dummyclassifier", "dummy", "dummyclassifier-aeon"],
    "dummyclassifier-tsml",
    "dummyclassifier-sklearn",
]
shapelet_based_classifiers = [
    "stc-2hour",
    ["shapelettransformclassifier", "stc"],
    ["rdstclassifier", "rdst"],
    ["randomshapeletforestclassifier", "randomshapeletforest", "rsf"],
    ["sastclassifier", "sast"],
    ["rsastclassifier", "rsast"],
    ["learningshapeletclassifier", "ls"],
]
vector_classifiers = [
    ["rotationforestclassifier", "rotationforest", "rotf"],
    ["ridgeclassifiercv", "ridgecv"],
    ["logisticregression", "logistic"],
]


def get_classifier_by_name(
    classifier_name,
    random_state=None,
    n_jobs=1,
    fit_contract=0,
    checkpoint=None,
    **kwargs,
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
    n_jobs: int, default=1
        The number of jobs to run in parallel for both classifier ``fit`` and
        ``predict`` if available. `-1` means using all processors.
    fit_contract: int, default=0
        Contract time in minutes for classifier ``fit`` if available.
    checkpoint: str or None, default=None
        Path to a checkpoint file to save the classifier if available. No checkpointing
        if None.
    **kwargs
        Additional keyword arguments to be passed to the classifier.

    Return
    ------
    classifier : A BaseClassifier.
        The classifier matching the input classifier name.
    """
    c = classifier_name.casefold()

    if str_in_nested_list(convolution_based_classifiers, c):
        return _set_classifier_convolution_based(
            c, random_state, n_jobs, fit_contract, checkpoint, kwargs
        )
    elif str_in_nested_list(deep_learning_classifiers, c):
        return _set_classifier_deep_learning(
            c, random_state, n_jobs, fit_contract, checkpoint, kwargs
        )
    elif str_in_nested_list(dictionary_based_classifiers, c):
        return _set_classifier_dictionary_based(
            c, random_state, n_jobs, fit_contract, checkpoint, kwargs
        )
    elif str_in_nested_list(distance_based_classifiers, c):
        return _set_classifier_distance_based(
            c, random_state, n_jobs, fit_contract, checkpoint, kwargs
        )
    elif str_in_nested_list(feature_based_classifiers, c):
        return _set_classifier_feature_based(
            c, random_state, n_jobs, fit_contract, checkpoint, kwargs
        )
    elif str_in_nested_list(hybrid_classifiers, c):
        return _set_classifier_hybrid(
            c, random_state, n_jobs, fit_contract, checkpoint, kwargs
        )
    elif str_in_nested_list(interval_based_classifiers, c):
        return _set_classifier_interval_based(
            c, random_state, n_jobs, fit_contract, checkpoint, kwargs
        )
    elif str_in_nested_list(other_classifiers, c):
        return _set_classifier_other(
            c, random_state, n_jobs, fit_contract, checkpoint, kwargs
        )
    elif str_in_nested_list(shapelet_based_classifiers, c):
        return _set_classifier_shapelet_based(
            c, random_state, n_jobs, fit_contract, checkpoint, kwargs
        )
    elif str_in_nested_list(vector_classifiers, c):
        return _set_classifier_vector(
            c, random_state, n_jobs, fit_contract, checkpoint, kwargs
        )
    else:
        raise ValueError(f"UNKNOWN CLASSIFIER: {c} in get_classifier_by_name")


def _set_classifier_convolution_based(
    c, random_state, n_jobs, fit_contract, checkpoint, kwargs
):
    if c == "rocketclassifier" or c == "rocket":
        from aeon.classification.convolution_based import RocketClassifier

        return RocketClassifier(random_state=random_state, n_jobs=n_jobs, **kwargs)
    elif c == "minirocket" or c == "mini-rocket" or c == "minirocketclassifier":
        from aeon.classification.convolution_based import MiniRocketClassifier

        return MiniRocketClassifier(
            random_state=random_state,
            n_jobs=n_jobs,
            **kwargs,
        )
    elif c == "multirocket" or c == "multi-rocket" or c == "multirocketclassifier":
        from aeon.classification.convolution_based import MultiRocketClassifier

        return MultiRocketClassifier(
            random_state=random_state,
            n_jobs=n_jobs,
            **kwargs,
        )
    elif c == "arsenalclassifier" or c == "arsenal":
        from aeon.classification.convolution_based import Arsenal

        return Arsenal(
            random_state=random_state,
            n_jobs=n_jobs,
            time_limit_in_minutes=fit_contract,
            **kwargs,
        )
    elif c == "miniarsenal" or c == "mini-arsenal":
        from aeon.classification.convolution_based import Arsenal

        return Arsenal(
            rocket_transform="minirocket",
            random_state=random_state,
            n_jobs=n_jobs,
            time_limit_in_minutes=fit_contract,
            **kwargs,
        )
    elif c == "multiarsenal" or c == "multi-arsenal":
        from aeon.classification.convolution_based import Arsenal

        return Arsenal(
            rocket_transform="multirocket",
            random_state=random_state,
            n_jobs=n_jobs,
            time_limit_in_minutes=fit_contract,
            **kwargs,
        )
    elif c == "hydraclassifier" or c == "hydra":
        from aeon.classification.convolution_based import HydraClassifier

        return HydraClassifier(random_state=random_state, n_jobs=n_jobs, **kwargs)
    elif (
        c == "multirockethydraclassifier"
        or c == "multirockethydra"
        or c == "multirocket-hydra"
        or c == "mrhydra"
    ):
        from aeon.classification.convolution_based import MultiRocketHydraClassifier

        return MultiRocketHydraClassifier(
            random_state=random_state, n_jobs=n_jobs, **kwargs
        )


def _set_classifier_deep_learning(
    c, random_state, n_jobs, fit_contract, checkpoint, kwargs
):
    if c == "timecnnclassifier" or c == "timecnn" or c == "cnnclassifier" or c == "cnn":
        from aeon.classification.deep_learning import TimeCNNClassifier

        return TimeCNNClassifier(random_state=random_state, **kwargs)
    elif c == "fcnclassifier" or c == "fcnn":
        from aeon.classification.deep_learning import FCNClassifier

        return FCNClassifier(random_state=random_state, **kwargs)
    elif c == "mlpclassifier" or c == "mlp":
        from aeon.classification.deep_learning import MLPClassifier

        return MLPClassifier(random_state=random_state, **kwargs)
    elif c == "encoderclassifier" or c == "encoder":
        from aeon.classification.deep_learning import EncoderClassifier

        return EncoderClassifier(random_state=random_state, **kwargs)
    elif c == "resnetclassifier" or c == "resnet":
        from aeon.classification.deep_learning import ResNetClassifier

        return ResNetClassifier(random_state=random_state, **kwargs)
    elif c == "individualinceptionclassifier" or c == "singleinception":
        from aeon.classification.deep_learning import IndividualInceptionClassifier

        return IndividualInceptionClassifier(random_state=random_state, **kwargs)
    elif c == "inceptiontimeclassifier" or c == "inceptiontime":
        from aeon.classification.deep_learning import InceptionTimeClassifier

        return InceptionTimeClassifier(random_state=random_state, **kwargs)
    elif c == "h-inceptiontimeclassifier" or c == "h-inceptiontime":
        from aeon.classification.deep_learning import InceptionTimeClassifier

        return InceptionTimeClassifier(
            use_custom_filters=True, random_state=random_state, **kwargs
        )
    elif c == "litetimeclassifier" or c == "litetime":
        from aeon.classification.deep_learning import LITETimeClassifier

        return LITETimeClassifier(random_state=random_state, **kwargs)
    elif c == "litetime-mv":
        from aeon.classification.deep_learning import LITETimeClassifier

        return LITETimeClassifier(use_litemv=True, random_state=random_state, **kwargs)
    elif c == "individualliteclassifier" or c == "individuallite":
        from aeon.classification.deep_learning import IndividualLITEClassifier

        return IndividualLITEClassifier(random_state=random_state, **kwargs)
    elif c == "disjointcnnclassifier" or c == "disjointcnn":
        from aeon.classification.deep_learning import DisjointCNNClassifier

        return DisjointCNNClassifier(random_state=random_state, **kwargs)


def _set_classifier_dictionary_based(
    c, random_state, n_jobs, fit_contract, checkpoint, kwargs
):
    if c == "bossensemble" or c == "boss":
        from aeon.classification.dictionary_based import BOSSEnsemble

        return BOSSEnsemble(
            random_state=random_state,
            n_jobs=n_jobs,
            **kwargs,
        )
    elif c == "individualboss":
        from aeon.classification.dictionary_based import IndividualBOSS

        return IndividualBOSS(
            random_state=random_state,
            n_jobs=n_jobs,
            **kwargs,
        )
    elif c == "contractableboss" or c == "cboss":
        from aeon.classification.dictionary_based import ContractableBOSS

        return ContractableBOSS(
            random_state=random_state,
            n_jobs=n_jobs,
            time_limit_in_minutes=fit_contract,
            **kwargs,
        )
    elif c == "temporaldictionaryensemble" or c == "tde":
        from aeon.classification.dictionary_based import TemporalDictionaryEnsemble

        return TemporalDictionaryEnsemble(
            random_state=random_state,
            n_jobs=n_jobs,
            time_limit_in_minutes=fit_contract,
            **kwargs,
        )
    elif c == "individualtde":
        from aeon.classification.dictionary_based import IndividualTDE

        return IndividualTDE(random_state=random_state, n_jobs=n_jobs, **kwargs)
    elif c == "weasel":
        from aeon.classification.dictionary_based import WEASEL

        return WEASEL(random_state=random_state, n_jobs=n_jobs, **kwargs)
    elif c == "weasel-logistic":
        from aeon.classification.dictionary_based import WEASEL

        return WEASEL(
            random_state=random_state,
            n_jobs=n_jobs,
            support_probabilities=True,
            **kwargs,
        )
    elif c == "muse":
        from aeon.classification.dictionary_based import MUSE

        return MUSE(random_state=random_state, n_jobs=n_jobs, **kwargs)
    elif c == "muse-logistic":
        from aeon.classification.dictionary_based import MUSE

        return MUSE(
            random_state=random_state,
            n_jobs=n_jobs,
            support_probabilities=True,
            **kwargs,
        )
    elif (
        c == "weasel_v2"
        or c == "weaseldilation"
        or c == "weasel-dilation"
        or c == "weasel-d"
    ):
        from aeon.classification.dictionary_based import WEASEL_V2

        return WEASEL_V2(
            random_state=random_state,
            n_jobs=n_jobs,
            **kwargs,
        )
    elif c == "redcomets":
        from aeon.classification.dictionary_based import REDCOMETS

        return REDCOMETS(random_state=random_state, n_jobs=n_jobs, **kwargs)
    elif c == "redcomets-500":
        from aeon.classification.dictionary_based import REDCOMETS

        return REDCOMETS(
            n_trees=500, random_state=random_state, n_jobs=n_jobs, **kwargs
        )
    elif c == "mrseqlclassifier" or c == "mrseql":
        from aeon.classification.dictionary_based import MrSEQLClassifier

        return MrSEQLClassifier(**kwargs)
    elif c == "mrsqmclassifier" or c == "mrsqm":
        from aeon.classification.dictionary_based import MrSQMClassifier

        return MrSQMClassifier(random_state=random_state, **kwargs)


def _set_classifier_distance_based(
    c, random_state, n_jobs, fit_contract, checkpoint, kwargs
):
    if c == "kneighborstimeseriesclassifier" or c == "dtw" or c == "1nn-dtw":
        from aeon.classification.distance_based import KNeighborsTimeSeriesClassifier

        return KNeighborsTimeSeriesClassifier(distance="dtw", n_jobs=n_jobs, **kwargs)
    elif c == "ed" or c == "1nn-euclidean" or c == "1nn-ed":
        from aeon.classification.distance_based import KNeighborsTimeSeriesClassifier

        return KNeighborsTimeSeriesClassifier(
            distance="euclidean", n_jobs=n_jobs, **kwargs
        )
    elif c == "msm" or c == "1nn-msm":
        from aeon.classification.distance_based import KNeighborsTimeSeriesClassifier

        return KNeighborsTimeSeriesClassifier(distance="msm", n_jobs=n_jobs, **kwargs)
    elif c == "twe" or c == "1nn-twe":
        from aeon.classification.distance_based import KNeighborsTimeSeriesClassifier

        return KNeighborsTimeSeriesClassifier(distance="twe", n_jobs=n_jobs, **kwargs)
    elif c == "elasticensemble" or c == "ee":
        from aeon.classification.distance_based import ElasticEnsemble

        return ElasticEnsemble(random_state=random_state, n_jobs=n_jobs, **kwargs)
    elif c == "1nn-dtw-cv":
        from aeon.classification.distance_based import KNeighborsTimeSeriesClassifier
        from sklearn.model_selection import GridSearchCV

        param_grid = {"distance_params": [{"window": x / 100} for x in range(0, 100)]}
        return GridSearchCV(
            estimator=KNeighborsTimeSeriesClassifier(),
            param_grid=param_grid,
            scoring="accuracy",
            **kwargs,
        )
    elif c == "grailclassifier" or c == "grail":
        from tsml.distance_based import GRAILClassifier

        return GRAILClassifier(**kwargs)
    elif c == "proximitytree" or c == "proximitytreeclassifier":
        from aeon.classification.distance_based import ProximityTree

        return ProximityTree(random_state=random_state, **kwargs)

    elif c == "proximityforest" or c == "pf":
        from aeon.classification.distance_based import ProximityForest

        return ProximityForest(random_state=random_state, n_jobs=n_jobs, **kwargs)


def _set_classifier_feature_based(
    c, random_state, n_jobs, fit_contract, checkpoint, kwargs
):
    if c == "summary-500":
        from aeon.classification.feature_based import SummaryClassifier
        from sklearn.ensemble import RandomForestClassifier

        return SummaryClassifier(
            estimator=RandomForestClassifier(n_estimators=500),
            random_state=random_state,
            n_jobs=n_jobs,
            **kwargs,
        )
    elif c == "summaryclassifier" or c == "summary":
        from aeon.classification.feature_based import SummaryClassifier

        return SummaryClassifier(random_state=random_state, n_jobs=n_jobs, **kwargs)
    elif c == "catch22-500":
        from aeon.classification.feature_based import Catch22Classifier
        from sklearn.ensemble import RandomForestClassifier

        return Catch22Classifier(
            estimator=RandomForestClassifier(n_estimators=500),
            outlier_norm=False,
            random_state=random_state,
            n_jobs=n_jobs,
            **kwargs,
        )
    elif c == "catch22classifier" or c == "catch22":
        from aeon.classification.feature_based import Catch22Classifier

        return Catch22Classifier(
            outlier_norm=False, random_state=random_state, n_jobs=n_jobs, **kwargs
        )
    elif c == "catch22-outlier":
        from aeon.classification.feature_based import Catch22Classifier

        return Catch22Classifier(
            outlier_norm=True, random_state=random_state, n_jobs=n_jobs, **kwargs
        )
    elif c == "freshprinceclassifier" or c == "freshprince":
        from aeon.classification.feature_based import FreshPRINCEClassifier

        return FreshPRINCEClassifier(random_state=random_state, n_jobs=n_jobs, **kwargs)
    elif c == "freshprince-500":
        from aeon.classification.feature_based import FreshPRINCEClassifier

        return FreshPRINCEClassifier(
            n_estimators=500, random_state=random_state, n_jobs=n_jobs, **kwargs
        )
    elif c == "tsfresh-nofs":
        from aeon.classification.feature_based import TSFreshClassifier

        return TSFreshClassifier(
            relevant_feature_extractor=False,
            random_state=random_state,
            n_jobs=n_jobs,
            **kwargs,
        )
    elif c == "tsfreshclassifier" or c == "tsfresh":
        from aeon.classification.feature_based import TSFreshClassifier

        return TSFreshClassifier(random_state=random_state, n_jobs=n_jobs, **kwargs)
    elif c == "signatureclassifier" or c == "signatures":
        from aeon.classification.feature_based import SignatureClassifier

        return SignatureClassifier(random_state=random_state, **kwargs)


def _set_classifier_hybrid(c, random_state, n_jobs, fit_contract, checkpoint, kwargs):
    if c == "hivecotev1" or c == "hc1":
        from aeon.classification.hybrid import HIVECOTEV1

        return HIVECOTEV1(random_state=random_state, n_jobs=n_jobs, **kwargs)
    elif c == "hivecotev2" or c == "hc2":
        from aeon.classification.hybrid import HIVECOTEV2

        return HIVECOTEV2(
            random_state=random_state,
            n_jobs=n_jobs,
            time_limit_in_minutes=fit_contract,
            **kwargs,
        )
    elif c == "ristclassifier" or c == "rist" or c == "rist-extrat":
        from aeon.classification.hybrid import RISTClassifier
        from sklearn.ensemble import ExtraTreesClassifier

        return RISTClassifier(
            random_state=random_state,
            n_jobs=n_jobs,
            estimator=ExtraTreesClassifier(n_estimators=500, criterion="entropy"),
            **kwargs,
        )


def _set_classifier_interval_based(
    c, random_state, n_jobs, fit_contract, checkpoint, kwargs
):
    if c == "rstsf-500":
        from aeon.classification.interval_based import RSTSF

        return RSTSF(
            n_estimators=500, random_state=random_state, n_jobs=n_jobs, **kwargs
        )
    elif c == "rstsfclassifier" or c == "rstsf" or c == "r-stsf":
        from aeon.classification.interval_based import RSTSF

        return RSTSF(random_state=random_state, n_jobs=n_jobs, **kwargs)
    elif c == "rise-500":
        from aeon.classification.interval_based import (
            RandomIntervalSpectralEnsembleClassifier,
        )

        return RandomIntervalSpectralEnsembleClassifier(
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
            random_state=random_state,
            n_jobs=n_jobs,
            **kwargs,
        )
    elif c == "tsf-500":
        from aeon.classification.interval_based import TimeSeriesForestClassifier

        return TimeSeriesForestClassifier(
            n_estimators=500,
            random_state=random_state,
            n_jobs=n_jobs,
            **kwargs,
        )
    elif c == "timeseriesforestclassifier" or c == "tsf":
        from aeon.classification.interval_based import TimeSeriesForestClassifier

        return TimeSeriesForestClassifier(
            random_state=random_state,
            n_jobs=n_jobs,
            **kwargs,
        )
    elif c == "cif-500":
        from aeon.classification.interval_based import CanonicalIntervalForestClassifier

        return CanonicalIntervalForestClassifier(
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
    elif c == "stsf-500":
        from aeon.classification.interval_based import SupervisedTimeSeriesForest

        return SupervisedTimeSeriesForest(
            n_estimators=500,
            random_state=random_state,
            n_jobs=n_jobs,
            **kwargs,
        )
    elif c == "supervisedtimeseriesforest" or c == "stsf":
        from aeon.classification.interval_based import SupervisedTimeSeriesForest

        return SupervisedTimeSeriesForest(
            random_state=random_state,
            n_jobs=n_jobs,
            **kwargs,
        )
    elif c == "drcif-500":
        from aeon.classification.interval_based import DrCIFClassifier

        return DrCIFClassifier(
            n_estimators=500,
            random_state=random_state,
            n_jobs=n_jobs,
            time_limit_in_minutes=fit_contract,
            **kwargs,
        )
    elif c == "drcif" or c == "drcifclassifier":
        from aeon.classification.interval_based import DrCIFClassifier

        return DrCIFClassifier(
            random_state=random_state,
            n_jobs=n_jobs,
            time_limit_in_minutes=fit_contract,
            **kwargs,
        )
    elif c == "summary-intervals":
        from aeon.classification.interval_based import RandomIntervalClassifier
        from aeon.transformations.collection.feature_based import SevenNumberSummary
        from sklearn.ensemble import RandomForestClassifier

        return RandomIntervalClassifier(
            features=SevenNumberSummary(),
            estimator=RandomForestClassifier(n_estimators=500),
            random_state=random_state,
            n_jobs=n_jobs,
            **kwargs,
        )
    elif c == "randomintervals-500" or c == "catch22-intervals-500":
        from aeon.classification.interval_based import RandomIntervalClassifier
        from sklearn.ensemble import RandomForestClassifier

        return RandomIntervalClassifier(
            estimator=RandomForestClassifier(n_estimators=500),
            random_state=random_state,
            n_jobs=n_jobs,
            **kwargs,
        )
    elif (
        c == "randomintervalclassifier"
        or c == "randomintervals"
        or c == "catch22-intervals"
    ):
        from aeon.classification.interval_based import RandomIntervalClassifier

        return RandomIntervalClassifier(
            random_state=random_state, n_jobs=n_jobs, **kwargs
        )
    elif c == "supervisedintervalclassifier" or c == "supervisedintervals":
        from aeon.classification.interval_based import SupervisedIntervalClassifier

        return SupervisedIntervalClassifier(
            random_state=random_state, n_jobs=n_jobs, **kwargs
        )
    elif c == "quantclassifier" or c == "quant":
        from aeon.classification.interval_based import QUANTClassifier

        return QUANTClassifier(random_state=random_state, **kwargs)


def _set_classifier_other(c, random_state, n_jobs, fit_contract, checkpoint, kwargs):
    if c == "dummyclassifier" or c == "dummy" or c == "dummyclassifier-aeon":
        from aeon.classification import DummyClassifier

        return DummyClassifier(random_state=random_state, **kwargs)
    elif c == "dummyclassifier-tsml":
        from tsml.dummy import DummyClassifier

        return DummyClassifier(random_state=random_state, **kwargs)
    elif c == "dummyclassifier-sklearn":
        from sklearn.dummy import DummyClassifier

        return DummyClassifier(random_state=random_state, **kwargs)


def _set_classifier_shapelet_based(
    c, random_state, n_jobs, fit_contract, checkpoint, kwargs
):
    if c == "stc-2hour":
        from aeon.classification.shapelet_based import ShapeletTransformClassifier

        return ShapeletTransformClassifier(
            transform_limit_in_minutes=120,
            random_state=random_state,
            n_jobs=n_jobs,
            **kwargs,
        )
    elif c == "shapelettransformclassifier" or c == "stc":
        from aeon.classification.shapelet_based import ShapeletTransformClassifier

        return ShapeletTransformClassifier(
            random_state=random_state,
            n_jobs=n_jobs,
            time_limit_in_minutes=fit_contract,
            **kwargs,
        )
    elif c == "rdstclassifier" or c == "rdst":
        from aeon.classification.shapelet_based import RDSTClassifier

        return RDSTClassifier(random_state=random_state, **kwargs)
    elif (
        c == "randomshapeletforestclassifier"
        or c == "randomshapeletforest"
        or c == "rsf"
    ):
        from tsml.shapelet_based import RandomShapeletForestClassifier

        return RandomShapeletForestClassifier(
            random_state=random_state, n_jobs=n_jobs, **kwargs
        )
    elif c == "sastclassifier" or c == "sast":
        from aeon.classification.shapelet_based import SASTClassifier

        return SASTClassifier(seed=random_state, n_jobs=n_jobs, **kwargs)
    elif c == "rsastclassifier" or c == "rsast":
        from aeon.classification.shapelet_based import RSASTClassifier

        return RSASTClassifier(seed=random_state, n_jobs=n_jobs, **kwargs)
    elif c == "learningshapeletclassifier" or c == "ls":
        from aeon.classification.shapelet_based import LearningShapeletClassifier

        return LearningShapeletClassifier(random_state=random_state, **kwargs)


def _set_classifier_vector(c, random_state, n_jobs, fit_contract, checkpoint, kwargs):
    if c == "rotationforestclassifier" or c == "rotationforest" or c == "rotf":
        from aeon.classification.sklearn import RotationForestClassifier

        return RotationForestClassifier(
            random_state=random_state,
            n_jobs=n_jobs,
            time_limit_in_minutes=fit_contract,
            **kwargs,
        )
    elif c == "ridgeclassifiercv" or c == "ridgecv":
        from sklearn.linear_model import RidgeClassifierCV

        return RidgeClassifierCV(**kwargs)
    elif c == "logisticregression" or c == "logistic":
        from sklearn.linear_model import LogisticRegression

        return LogisticRegression(random_state=random_state, n_jobs=n_jobs, **kwargs)
