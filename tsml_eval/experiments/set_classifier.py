"""Set classifier function."""

__author__ = ["TonyBagnall", "MatthewMiddlehurst"]

from tsml_eval.utils.functions import str_in_nested_list

convolution_based_classifiers = [
    ["RocketClassifier", "rocket"],
    ["minirocket", "mini-rocket"],
    ["multirocket", "multi-rocket"],
    ["arsenalclassifier", "Arsenal"],
    ["miniarsenal", "mini-arsenal"],
    ["multiarsenal", "multi-arsenal"],
    "HYDRA",
    ["HydraMultiRocket", "hydra-multirocket"],
]
deep_learning_classifiers = [
    ["CNNClassifier", "cnn"],
    ["FCNClassifier", "fcnn"],
    ["MLPClassifier", "mlp"],
    ["TapNetClassifier", "tapnet"],
    ["ResNetClassifier", "resnet"],
    ["IndividualInceptionClassifier", "singleinception"],
    ["InceptionTimeClassifier", "inceptiontime"],
]
dictionary_based_classifiers = [
    ["BOSSEnsemble", "boss"],
    "IndividualBOSS",
    ["ContractableBOSS", "cboss"],
    ["TemporalDictionaryEnsemble", "tde"],
    "IndividualTDE",
    "WEASEL",
    "weasel-logistic",
    "MUSE",
    "muse-logistic",
    ["WEASEL_V2", "weaseldilation", "weasel-dilation", "weasel-d"],
    ["MUSEDilation", "muse-dilation", "muse-d"],
]
distance_based_classifiers = [
    ["KNeighborsTimeSeriesClassifier", "dtw", "1nn-dtw"],
    ["ed", "1nn-euclidean", "1nn-ed"],
    ["msm", "1nn-msm"],
    ["twe", "1nn-twe"],
    "1nn-dtw-cv",
    ["ElasticEnsemble", "ee"],
    "ShapeDTW",
    ["MatrixProfileClassifier", "matrixprofile"],
]
feature_based_classifiers = [
    "summary-500",
    ["SummaryClassifier", "summary"],
    "catch22-500",
    ["Catch22Classifier", "catch22"],
    ["FreshPRINCEClassifier", "freshprince"],
    "tsfresh-nofs",
    ["TSFreshClassifier", "tsfresh"],
    ["SignatureClassifier", "signatures"],
]
hybrid_classifiers = [
    ["HIVECOTEV1", "hc1"],
    ["HIVECOTEV2", "hc2"],
    ["TsChief", "ts-chief"],
]
interval_based_classifiers = [
    "rstsf-500",
    ["RSTSFClassifier", "rstsf", "r-stsf"],
    "rise-500",
    ["RandomIntervalSpectralEnsemble", "rise"],
    "tsf-500",
    ["TimeSeriesForestClassifier", "tsf"],
    "cif-500",
    ["CanonicalIntervalForest", "cif"],
    "stsf-500",
    ["SupervisedTimeSeriesForest", "stsf"],
    "drcif-500",
    "DrCIF",
    "summary-intervals",
    ["randomintervals-rf", "catch22-intervals-rf"],
    ["RandomIntervalClassifier", "randomintervals", "catch22-intervals"],
]
other_classifiers = [
    ["DummyClassifier", "dummy", "dummyclassifier-aeon"],
    "dummyclassifier-tsml",
    "dummyclassifier-sklearn",
]
shapelet_based_classifiers = [
    "stc-2hour",
    ["ShapeletTransformClassifier", "stc"],
    "RDST",
    ["RDSTEnsemble", "rdst-ensemble"],
    ["RandomShapeletForestClassifier", "randomshapeletforest", "rsf"],
    ["MrSQMClassifier", "mrsqm"],
]
vector_classifiers = [
    ["RotationForestClassifier", "rotationforest", "rotf"],
]


def set_classifier(
    classifier_name,
    random_state=None,
    n_jobs=1,
    build_train_file=False,
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
    c = classifier_name.casefold()

    if str_in_nested_list(convolution_based_classifiers, c):
        return _set_classifier_convolution_based(
            c, random_state, n_jobs, build_train_file, fit_contract, checkpoint, kwargs
        )
    elif str_in_nested_list(deep_learning_classifiers, c):
        return _set_classifier_deep_learning(
            c, random_state, n_jobs, build_train_file, fit_contract, checkpoint, kwargs
        )
    elif str_in_nested_list(dictionary_based_classifiers, c):
        return _set_classifier_dictionary_based(
            c, random_state, n_jobs, build_train_file, fit_contract, checkpoint, kwargs
        )
    elif str_in_nested_list(distance_based_classifiers, c):
        return _set_classifier_distance_based(
            c, random_state, n_jobs, build_train_file, fit_contract, checkpoint, kwargs
        )
    elif str_in_nested_list(feature_based_classifiers, c):
        return _set_classifier_feature_based(
            c, random_state, n_jobs, build_train_file, fit_contract, checkpoint, kwargs
        )
    elif str_in_nested_list(hybrid_classifiers, c):
        return _set_classifier_hybrid(
            c, random_state, n_jobs, build_train_file, fit_contract, checkpoint, kwargs
        )
    elif str_in_nested_list(interval_based_classifiers, c):
        return _set_classifier_interval_based(
            c, random_state, n_jobs, build_train_file, fit_contract, checkpoint, kwargs
        )
    elif str_in_nested_list(other_classifiers, c):
        return _set_classifier_other(
            c, random_state, n_jobs, build_train_file, fit_contract, checkpoint, kwargs
        )
    elif str_in_nested_list(shapelet_based_classifiers, c):
        return _set_classifier_shapelet_based(
            c, random_state, n_jobs, build_train_file, fit_contract, checkpoint, kwargs
        )
    elif str_in_nested_list(vector_classifiers, c):
        return _set_classifier_vector(
            c, random_state, n_jobs, build_train_file, fit_contract, checkpoint, kwargs
        )
    else:
        raise ValueError(f"UNKNOWN CLASSIFIER {c} in set_classifier")


def _set_classifier_convolution_based(
    c, random_state, n_jobs, build_train_file, fit_contract, checkpoint, kwargs
):
    if c == "rocketclassifier" or c == "rocket":
        from aeon.classification.convolution_based import RocketClassifier

        return RocketClassifier(random_state=random_state, n_jobs=n_jobs, **kwargs)
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
            save_transformed_data=build_train_file,
            time_limit_in_minutes=fit_contract,
            **kwargs,
        )
    elif c == "miniarsenal" or c == "mini-arsenal":
        from aeon.classification.convolution_based import Arsenal

        return Arsenal(
            rocket_transform="minirocket",
            random_state=random_state,
            n_jobs=n_jobs,
            save_transformed_data=build_train_file,
            time_limit_in_minutes=fit_contract,
            **kwargs,
        )
    elif c == "multiarsenal" or c == "multi-arsenal":
        from aeon.classification.convolution_based import Arsenal

        return Arsenal(
            rocket_transform="multirocket",
            random_state=random_state,
            n_jobs=n_jobs,
            save_transformed_data=build_train_file,
            time_limit_in_minutes=fit_contract,
            **kwargs,
        )
    elif c == "hydra":
        from tsml_eval.estimators.classification.convolution_based.hydra import HYDRA

        return HYDRA(random_state=random_state, n_jobs=n_jobs, **kwargs)
    elif c == "hydramultirocket" or c == "hydra-multirocket":
        from tsml_eval.estimators.classification.convolution_based.hydra import (
            HydraMultiRocket,
        )

        return HydraMultiRocket(random_state=random_state, n_jobs=n_jobs, **kwargs)


def _set_classifier_deep_learning(
    c, random_state, n_jobs, build_train_file, fit_contract, checkpoint, kwargs
):
    if c == "cnnclassifier" or c == "cnn":
        from aeon.classification.deep_learning import CNNClassifier

        return CNNClassifier(random_state=random_state, **kwargs)
    elif c == "fcnclassifier" or c == "fcnn":
        from aeon.classification.deep_learning.fcn import FCNClassifier

        return FCNClassifier(random_state=random_state, **kwargs)
    elif c == "mlpclassifier" or c == "mlp":
        from aeon.classification.deep_learning.mlp import MLPClassifier

        return MLPClassifier(random_state=random_state, **kwargs)
    elif c == "tapnetclassifier" or c == "tapnet":
        from aeon.classification.deep_learning.tapnet import TapNetClassifier

        return TapNetClassifier(random_state=random_state, **kwargs)
    elif c == "resnetclassifier" or c == "resnet":
        from aeon.classification.deep_learning.resnet import ResNetClassifier

        return ResNetClassifier(random_state=random_state, **kwargs)
    elif c == "individualinceptionclassifier" or c == "singleinception":
        from aeon.classification.deep_learning.inception_time import (
            IndividualInceptionClassifier,
        )

        return IndividualInceptionClassifier(random_state=random_state, **kwargs)
    elif c == "inceptiontimeclassifier" or c == "inceptiontime":
        from aeon.classification.deep_learning.inception_time import (
            InceptionTimeClassifier,
        )

        return InceptionTimeClassifier(random_state=random_state, **kwargs)


def _set_classifier_dictionary_based(
    c, random_state, n_jobs, build_train_file, fit_contract, checkpoint, kwargs
):
    if c == "bossensemble" or c == "boss":
        from aeon.classification.dictionary_based import BOSSEnsemble

        return BOSSEnsemble(
            random_state=random_state,
            n_jobs=n_jobs,
            save_train_predictions=build_train_file,
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
            save_train_predictions=build_train_file,
            time_limit_in_minutes=fit_contract,
            **kwargs,
        )
    elif c == "temporaldictionaryensemble" or c == "tde":
        from aeon.classification.dictionary_based import TemporalDictionaryEnsemble

        return TemporalDictionaryEnsemble(
            random_state=random_state,
            save_train_predictions=build_train_file,
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

        return WEASEL_V2(random_state=random_state, n_jobs=n_jobs, **kwargs)
    elif c == "musedilation" or c == "muse-dilation" or c == "muse-d":
        from tsml_eval.estimators.classification.dictionary_based.muse import (
            MUSEDilation,
        )

        return MUSEDilation(random_state=random_state, n_jobs=n_jobs, **kwargs)


def _set_classifier_distance_based(
    c, random_state, n_jobs, build_train_file, fit_contract, checkpoint, kwargs
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
    elif c == "shapedtw":
        from aeon.classification.distance_based import ShapeDTW

        return ShapeDTW(**kwargs)
    elif c == "matrixprofileclassifier" or c == "matrixprofile":
        from aeon.classification.feature_based import MatrixProfileClassifier

        return MatrixProfileClassifier(
            random_state=random_state, n_jobs=n_jobs, **kwargs
        )
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


def _set_classifier_feature_based(
    c, random_state, n_jobs, build_train_file, fit_contract, checkpoint, kwargs
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
            random_state=random_state,
            n_jobs=n_jobs,
            **kwargs,
        )
    elif c == "catch22classifier" or c == "catch22":
        from aeon.classification.feature_based import Catch22Classifier

        return Catch22Classifier(random_state=random_state, n_jobs=n_jobs, **kwargs)
    elif c == "freshprinceclassifier" or c == "freshprince":
        from aeon.classification.feature_based import FreshPRINCEClassifier

        return FreshPRINCEClassifier(random_state=random_state, n_jobs=n_jobs, **kwargs)
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


def _set_classifier_hybrid(
    c, random_state, n_jobs, build_train_file, fit_contract, checkpoint, kwargs
):
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
    elif c == "tschief" or c == "ts-chief":
        from tsml_eval._wip.tschief.tschief import TsChief

        return TsChief(random_state=random_state, **kwargs)


def _set_classifier_interval_based(
    c, random_state, n_jobs, build_train_file, fit_contract, checkpoint, kwargs
):
    if c == "rstsf-500":
        from tsml.interval_based import RSTSFClassifier

        return RSTSFClassifier(
            n_estimators=500, random_state=random_state, n_jobs=n_jobs, **kwargs
        )
    elif c == "rstsfclassifier" or c == "rstsf" or c == "r-stsf":
        from tsml.interval_based import RSTSFClassifier

        return RSTSFClassifier(random_state=random_state, n_jobs=n_jobs, **kwargs)
    elif c == "rise-500":
        from aeon.classification.interval_based import RandomIntervalSpectralEnsemble

        return RandomIntervalSpectralEnsemble(
            n_estimators=500, random_state=random_state, n_jobs=n_jobs, **kwargs
        )
    elif c == "randomintervalspectralensemble" or c == "rise":
        from aeon.classification.interval_based import RandomIntervalSpectralEnsemble

        return RandomIntervalSpectralEnsemble(
            random_state=random_state, n_jobs=n_jobs, **kwargs
        )
    elif c == "tsf-500":
        from aeon.classification.interval_based import TimeSeriesForestClassifier

        return TimeSeriesForestClassifier(
            n_estimators=500, random_state=random_state, n_jobs=n_jobs, **kwargs
        )
    elif c == "timeseriesforestclassifier" or c == "tsf":
        from aeon.classification.interval_based import TimeSeriesForestClassifier

        return TimeSeriesForestClassifier(
            random_state=random_state, n_jobs=n_jobs, **kwargs
        )
    elif c == "cif-500":
        from aeon.classification.interval_based import CanonicalIntervalForest

        return CanonicalIntervalForest(
            n_estimators=500, random_state=random_state, n_jobs=n_jobs, **kwargs
        )
    elif c == "canonicalintervalforest" or c == "cif":
        from aeon.classification.interval_based import CanonicalIntervalForest

        return CanonicalIntervalForest(
            random_state=random_state, n_jobs=n_jobs, **kwargs
        )
    elif c == "stsf-500":
        from aeon.classification.interval_based import SupervisedTimeSeriesForest

        return SupervisedTimeSeriesForest(
            n_estimators=500, random_state=random_state, n_jobs=n_jobs, **kwargs
        )
    elif c == "supervisedtimeseriesforest" or c == "stsf":
        from aeon.classification.interval_based import SupervisedTimeSeriesForest

        return SupervisedTimeSeriesForest(
            random_state=random_state, n_jobs=n_jobs, **kwargs
        )
    elif c == "drcif-500":
        from aeon.classification.interval_based import DrCIF

        return DrCIF(
            n_estimators=500,
            random_state=random_state,
            save_transformed_data=build_train_file,
            n_jobs=n_jobs,
            time_limit_in_minutes=fit_contract,
            **kwargs,
        )
    elif c == "drcif":
        from aeon.classification.interval_based import DrCIF

        return DrCIF(
            random_state=random_state,
            save_transformed_data=build_train_file,
            n_jobs=n_jobs,
            time_limit_in_minutes=fit_contract,
            **kwargs,
        )
    elif c == "summary-intervals":
        from aeon.classification.interval_based import RandomIntervalClassifier
        from aeon.transformations.series.summarize import SummaryTransformer
        from sklearn.ensemble import RandomForestClassifier

        return RandomIntervalClassifier(
            interval_transformers=SummaryTransformer(
                summary_function=("mean", "std", "min", "max"),
                quantiles=(0.25, 0.5, 0.75),
            ),
            estimator=RandomForestClassifier(n_estimators=500),
            random_state=random_state,
            n_jobs=n_jobs,
            **kwargs,
        )
    elif c == "randomintervals-rf" or c == "catch22-intervals-rf":
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


def _set_classifier_other(
    c, random_state, n_jobs, build_train_file, fit_contract, checkpoint, kwargs
):
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
    c, random_state, n_jobs, build_train_file, fit_contract, checkpoint, kwargs
):
    if c == "stc-2hour":
        from aeon.classification.shapelet_based import ShapeletTransformClassifier

        return ShapeletTransformClassifier(
            transform_limit_in_minutes=120,
            random_state=random_state,
            save_transformed_data=build_train_file,
            n_jobs=n_jobs,
            **kwargs,
        )
    elif c == "shapelettransformclassifier" or c == "stc":
        from aeon.classification.shapelet_based import ShapeletTransformClassifier

        return ShapeletTransformClassifier(
            random_state=random_state,
            save_transformed_data=build_train_file,
            n_jobs=n_jobs,
            time_limit_in_minutes=fit_contract,
            **kwargs,
        )
    elif c == "rdst":
        from tsml_eval.estimators.classification.shapelet_based.rdst import RDST

        return RDST(random_state=random_state, **kwargs)
    elif c == "rdstensemble" or c == "rdst-ensemble":
        from tsml_eval.estimators.classification.shapelet_based.rdst import RDSTEnsemble

        return RDSTEnsemble(random_state=random_state, **kwargs)
    elif (
        c == "randomshapeletforestclassifier"
        or c == "randomshapeletforest"
        or c == "rsf"
    ):
        from tsml.shapelet_based import RandomShapeletForestClassifier

        return RandomShapeletForestClassifier(
            random_state=random_state, n_jobs=n_jobs, **kwargs
        )
    elif c == "mrsqmclassifier" or c == "mrsqm":
        from aeon.classification.shapelet_based import MrSQMClassifier

        return MrSQMClassifier(random_state=random_state, **kwargs)


def _set_classifier_vector(
    c, random_state, n_jobs, build_train_file, fit_contract, checkpoint, kwargs
):
    if c == "rotationforestclassifier" or c == "rotationforest" or c == "rotf":
        from tsml.vector import RotationForestClassifier

        return RotationForestClassifier(
            random_state=random_state,
            n_jobs=n_jobs,
            save_transformed_data=build_train_file,
            time_limit_in_minutes=fit_contract,
            **kwargs,
        )
