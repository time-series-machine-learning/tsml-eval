"""Get classifier function."""

__maintainer__ = ["TonyBagnall", "MatthewMiddlehurst"]

from tsml_eval.utils.functions import str_in_nested_list

convolution_based_classifiers = [
    ["rocketclassifier", "rocket"],
    "rocket-unequal",
    ["minirocket", "mini-rocket", "minirocketclassifier"],
    ["multirocket", "multi-rocket", "multirocketclassifier"],
    ["arsenalclassifier", "arsenal"],
    "arsenal-unequal",
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
    ["inceptiontimeclassifier", "inceptiontime", "it"],
    ["h-inceptiontimeclassifier", "h-inceptiontime"],
    ["litetimeclassifier", "litetime"],
    "litetime-mv",
    ["individualliteclassifier", "individuallite"],
    ["disjointcnnclassifier", "disjointcnn"],
]
# tsml-eval names for aeon-neuro's MOABBPipelineClassifier pipelines, named after
# MOABB's pipeline files (aeon_neuro...experimental.MOABB_PIPELINES)
moabb_pipelines = {
    "mdm-moabb": "MDM",
    "fgmdm-moabb": "FgMDM",
    "tssvm-moabb": "TSSVM",
    "tsen-moabb": "TSEN",
    "csp-lda-moabb": "CSP_LDA",
    "csp-svm-moabb": "CSP_SVM",
    "dlcsp-shlda-moabb": "DLCSPauto_shLDA",
    "trcsp-lda-moabb": "TRCSP_LDA",
    "fbcsp-svm-moabb": "FBCSP_SVM",
    "logvar-lda-moabb": "LogVar_LDA",
    "logvar-svm-moabb": "LogVar_SVM",
    "aug-tang-svm-moabb": "AUG_TANG_SVM",
    "xdawncov-mdm-moabb": "XdawnCov_MDM",
    "xdawncov-ts-svm-moabb": "XdawnCov_TS_SVM",
    "erpcov-mdm-moabb": "ERPCov_MDM",
    "xdawn-lda-moabb": "xDAWN_LDA",
}
# braindecode models trained with NeuralBench's recipe (NeuralBenchDeepClassifier),
# not aeon-neuro's own classifiers of the same models
neuralbench_deep_models = {
    "eegnet-nb": "EEGNet",
    "deep4net-nb": "Deep4Net",
    "eegconformer-nb": "EEGConformer",
    "shallowfbcspnet-nb": "ShallowFBCSPNet",
    # the rest of aeon-neuro's SUPPORTED_MODELS (_braindecode.py), the same recipe;
    # DGCNN and EEGSym are left out as they need channel_names
    "eeginceptionmi-nb": "EEGInceptionMI",
    "eegitnet-nb": "EEGITNet",
    "eegnex-nb": "EEGNeX",
    "eegsimpleconv-nb": "EEGSimpleConv",
    "eegtcnet-nb": "EEGTCNet",
    "sccnet-nb": "SCCNet",
    "tidnet-nb": "TIDNet",
    "sincshallownet-nb": "SincShallowNet",
    "fbcnet-nb": "FBCNet",
    "fbmsnet-nb": "FBMSNet",
    "fblightconvnet-nb": "FBLightConvNet",
    "ifnet-nb": "IFNet",
    "atcnet-nb": "ATCNet",
    "msvtnet-nb": "MSVTNet",
    "attentionbasenet-nb": "AttentionBaseNet",
    "sstdpn-nb": "SSTDPN",
    "mscformer-nb": "MSCFormer",
    "tmsanet-nb": "TMSANet",
    "tcformer-nb": "TCFormer",
    "gcnsnet-nb": "GCNsNet",
    "eeginceptionerp-nb": "EEGInceptionERP",
    "tsception-nb": "TSception",
    "eegminer-nb": "EEGMiner",
    "syncnet-nb": "SyncNet",
    "bdtcn-nb": "BDTCN",
    "sparcnet-nb": "SPARCNet",
    "brainmodule-nb": "BrainModule",
    "sleepstagerchambon2018-nb": "SleepStagerChambon2018",
    "contrawr-nb": "ContraWR",
}
neuro_classifiers = [
    ["eegnetclassifier", "eegnet"],
    ["deepconvnetclassifier", "deepconvnet", "dcn"],
    ["eegconformerclassifier", "eegconformer"],
    ["cbramodclassifier", "cbramod"],
    ["labramclassifier", "labram"],
    ["riemannianmdmclassifier", "riemannianmdm", "r-mdm"],
    ["riemannianknnclassifier", "riemannianknn", "r-knn"],
    ["riemannianmdm-oas", "r-mdm-oas"],
    ["riemannianknn-oas", "r-knn-oas"],
    ["riemannianmdm-lwf", "r-mdm-lwf"],
    ["riemannianknn-lwf", "r-knn-lwf"],
    ["cspclassifier", "csp", "csplda"],
    ["csprandomforestclassifier", "csprandomforest", "csprf"],
    ["cspsvmclassifier", "cspsvm", "csp-svm"],
    "csp+lda",
    "csp+svm",
    ["xdawnclassifier", "xdawn", "xdawnlda"],
    ["neuralbenchtslrclassifier", "tslr-nb"],
    ["neuralbenchxdawntslrclassifier", "xdawntslr-nb"],
    ["neuralbenchcospectralrclassifier", "cospectralr-nb"],
    ["moabbtslrclassifier", "tslr-moabb"],
    ["neuralbenchctnetclassifier", "ctnet-nb"],
    # standard EEG baselines: differential entropy (emotion/cognitive state) and
    # TRCA/template CCA (visual evoked potentials; run them band-passed, e.g.
    # TRCA-BP640, as slow drift swamps the raw signal)
    ["differentialentropyclassifier", "de-lr"],
    "de-svm",
    ["trcaclassifier", "trca"],
    # filter-bank TRCA (Nakanishi et al. 2018, 5 sub-bands); needs sfreq, filters
    # itself so no -BP suffix
    "fbtrca",
    # TRACE: Riemannian interval ensemble (aeon-neuro PR #296)
    ["trace", "traceclassifier"],
    ["templateccaclassifier", "templatecca"],
    *neuralbench_deep_models,
    *moabb_pipelines,
]
dictionary_based_classifiers = [
    ["bossensemble", "boss"],
    "individualboss",
    ["contractableboss", "cboss"],
    ["temporaldictionaryensemble", "tde"],
    "tde-unequal",
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
    ["proximityforest2", "pf2"],
]
feature_based_classifiers = [
    "summary-500",
    ["summaryclassifier", "summary"],
    "catch22-500",
    ["catch22classifier", "catch22"],
    "catch22-outlier",
    ["freshprinceclassifier", "freshprince"],
    "freshprince-unequal",
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
    "drcif-unequal",
    "summary-intervals",
    ["randomintervals-500", "catch22-intervals-500"],
    ["randomintervalclassifier", "randomintervals", "catch22-intervals"],
    ["supervisedintervalclassifier", "supervisedintervals"],
    ["quantclassifier", "quant"],
    "drcif-pipeline",
]
other_classifiers = [
    ["dummyclassifier", "dummy", "dummyclassifier-aeon"],
    "dummyclassifier-tsml",
    "dummyclassifier-sklearn",
]
shapelet_based_classifiers = [
    "stc-2hour",
    ["shapelettransformclassifier", "stc"],
    "stc-unequal",
    "stc-aeon",
    ["rdstclassifier", "rdst"],
    ["randomshapeletforestclassifier", "randomshapeletforest", "rsf"],
    ["sastclassifier", "sast"],
    ["rsastclassifier", "rsast"],
]
vector_classifiers = [
    ["rotationforestclassifier", "rotationforest", "rotf"],
    ["ridgeclassifiercv", "ridgecv"],
    ["logisticregression", "logistic"],
    ["svc", "svm", "supportvectorclassifier"],
    ["randomforestclassifier", "randomforest", "rf"],
    ["prevalidatedridgeclassifier", "prevalidatedridge"],
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

    # -BP830, -BP140 (band-pass, needs sfreq) and -norm put a preprocessing step in
    # front of any classifier, e.g. RiemannianMDM-OAS-BP830 or CNN-norm
    from tsml_eval.estimators.classification._eeg_preprocessing import (
        preprocessed_classifier,
    )

    pipeline = preprocessed_classifier(
        c,
        lambda base, kw: get_classifier_by_name(
            base,
            random_state=random_state,
            n_jobs=n_jobs,
            fit_contract=fit_contract,
            checkpoint=checkpoint,
            **kw,
        ),
        random_state,
        kwargs,
    )
    if pipeline is not None:
        return pipeline

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
    elif str_in_nested_list(neuro_classifiers, c):
        return _set_classifier_neuro(c, random_state, n_jobs, kwargs)
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
    elif c.startswith("neuro:"):
        return _set_classifier_neuro_by_class_name(
            c.removeprefix("neuro:"), random_state, n_jobs, kwargs
        )
    else:
        raise ValueError(f"UNKNOWN CLASSIFIER: {c} in get_classifier_by_name")


def _set_classifier_convolution_based(
    c, random_state, n_jobs, fit_contract, checkpoint, kwargs
):
    if c == "rocketclassifier" or c == "rocket":
        from aeon.classification.convolution_based import RocketClassifier

        return RocketClassifier(random_state=random_state, n_jobs=n_jobs, **kwargs)
    elif c == "rocket-unequal":
        from tsml_eval._wip.unequal_length._rocket import RocketClassifier

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
    elif c == "arsenal-unequal":
        from tsml_eval._wip.unequal_length._arsenal4 import Arsenal

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
    else:
        raise ValueError(f"UNKNOWN CLASSIFIER: {c} in get_classifier_by_name")


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
    elif c in ("inceptiontimeclassifier", "inceptiontime", "it"):
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
    else:
        raise ValueError(f"UNKNOWN CLASSIFIER: {c} in get_classifier_by_name")


def _set_classifier_neuro(c, random_state, n_jobs, kwargs):
    if c in ("eegnetclassifier", "eegnet"):
        from aeon_neuro.classification.deep_learning import EEGNetClassifier

        return EEGNetClassifier(random_state=random_state, **kwargs)
    elif c in ("deepconvnetclassifier", "deepconvnet", "dcn"):
        from aeon_neuro.classification.deep_learning import DeepConvNetClassifier

        return DeepConvNetClassifier(random_state=random_state, **kwargs)
    elif c in ("eegconformerclassifier", "eegconformer"):
        from aeon_neuro.classification.deep_learning import EEGConformerClassifier

        return EEGConformerClassifier(random_state=random_state, **kwargs)
    elif c in ("cbramodclassifier", "cbramod"):
        from aeon_neuro.classification.deep_learning import CBraModClassifier

        return CBraModClassifier(random_state=random_state, **kwargs)
    elif c in ("labramclassifier", "labram"):
        from aeon_neuro.classification.deep_learning import LabramClassifier

        # ch_names can come from the command line as "FP1,FP2,...".
        if isinstance(kwargs.get("ch_names"), str):
            kwargs["ch_names"] = [n.strip() for n in kwargs["ch_names"].split(",")]
        return LabramClassifier(random_state=random_state, **kwargs)
    elif c in ("riemannianmdmclassifier", "riemannianmdm", "r-mdm"):
        from aeon_neuro.classification.distance_based import RiemannianMDMClassifier

        return RiemannianMDMClassifier(n_jobs=n_jobs, **kwargs)
    elif c in ("riemannianknnclassifier", "riemannianknn", "r-knn"):
        from aeon_neuro.classification.distance_based import RiemannianKNNClassifier

        return RiemannianKNNClassifier(n_jobs=n_jobs, **kwargs)
    elif c in (
        "riemannianmdm-oas",
        "r-mdm-oas",
        "riemannianknn-oas",
        "r-knn-oas",
        "riemannianmdm-lwf",
        "r-mdm-lwf",
        "riemannianknn-lwf",
        "r-knn-lwf",
    ):
        # Shrinkage covariances, which stay positive definite where the sample
        # covariance ("scm") is singular. -OAS is oracle approximating shrinkage, as
        # MOABB's MDM pipelines use; its shrinkage is derived assuming Gaussian data.
        # -LWF is Ledoit-Wolf, which makes no such assumption (TRACE uses it).
        from aeon_neuro.classification.distance_based import (
            RiemannianKNNClassifier,
            RiemannianMDMClassifier,
        )

        cls = RiemannianMDMClassifier if "mdm" in c else RiemannianKNNClassifier
        est = "lwf" if c.endswith("-lwf") else "oas"
        return cls(n_jobs=n_jobs, **{"covariance_estimator": est, **kwargs})
    elif c in (
        "cspclassifier",
        "csp",
        "csplda",
        "csprandomforestclassifier",
        "csprandomforest",
        "csprf",
        "cspsvmclassifier",
        "cspsvm",
        "csp-svm",
    ):
        from aeon_neuro.classification.feature_based import CSPClassifier

        if c in ("csprandomforestclassifier", "csprandomforest", "csprf"):
            from sklearn.ensemble import RandomForestClassifier

            classifier = RandomForestClassifier(
                random_state=random_state, n_jobs=n_jobs
            )
        elif c in ("cspsvmclassifier", "cspsvm", "csp-svm"):
            from sklearn.svm import SVC

            classifier = SVC(
                kernel="linear", probability=True, random_state=random_state
            )
        else:
            classifier = None
        kwargs.setdefault("classifier", classifier)
        return CSPClassifier(random_state=random_state, **kwargs)
    elif c in ("csp+lda", "csp+svm"):
        # The MOABB CSP pipelines (pipelines/CSP.yml and CSP_SVM_grid.yml): OAS
        # covariance per trial, averaged per class, six CSP filters, log-variance.
        from aeon_neuro.classification.feature_based import CSPClassifier

        if c == "csp+lda":
            from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

            classifier = LinearDiscriminantAnalysis()
        else:
            from sklearn.model_selection import GridSearchCV, StratifiedKFold
            from sklearn.svm import SVC

            # The search tunes the SVC on CSP features fitted to the whole training
            # set. MOABB also tunes the number of filters (2-8), fixed at 6 here.
            # probability=True so results files get probabilities; it does not
            # change the predictions, which come from the decision function.
            classifier = GridSearchCV(
                SVC(kernel="linear", probability=True, random_state=random_state),
                {"C": [0.5, 1, 1.5], "kernel": ["rbf", "linear"]},
                cv=StratifiedKFold(3, shuffle=True, random_state=random_state),
                n_jobs=n_jobs,
            )
        csp_kwargs = {
            "classifier": classifier,
            "n_components": 6,
            "reg": "oas",
            "cov_est": "epoch",
        }
        csp_kwargs.update(kwargs)
        return CSPClassifier(random_state=random_state, **csp_kwargs)
    elif c in ("xdawnclassifier", "xdawn", "xdawnlda"):
        from aeon_neuro.classification.feature_based import XDAWNClassifier

        return XDAWNClassifier(random_state=random_state, **kwargs)
    # NeuralBench and MOABB baselines, from aeon-neuro's experimental package (not
    # its public API). All need the dataset's sampling rate as sfreq.
    elif c in ("neuralbenchtslrclassifier", "tslr-nb"):
        from aeon_neuro.classification.experimental import NeuralBenchTSLRClassifier

        return NeuralBenchTSLRClassifier(
            n_jobs=n_jobs, random_state=random_state, **kwargs
        )
    elif c in ("neuralbenchxdawntslrclassifier", "xdawntslr-nb"):
        from aeon_neuro.classification.experimental import (
            NeuralBenchXdawnTSLRClassifier,
        )

        return NeuralBenchXdawnTSLRClassifier(
            n_jobs=n_jobs, random_state=random_state, **kwargs
        )
    elif c in ("neuralbenchcospectralrclassifier", "cospectralr-nb"):
        from aeon_neuro.classification.experimental import (
            NeuralBenchCoSpectraLRClassifier,
        )

        return NeuralBenchCoSpectraLRClassifier(
            n_jobs=n_jobs, random_state=random_state, **kwargs
        )
    elif c in ("moabbtslrclassifier", "tslr-moabb"):
        from aeon_neuro.classification.experimental import MOABBTSLRClassifier

        return MOABBTSLRClassifier(random_state=random_state, **kwargs)
    elif c in ("neuralbenchctnetclassifier", "ctnet-nb"):
        from aeon_neuro.classification.experimental import NeuralBenchCTNetClassifier

        _read_train_groups(kwargs)
        return NeuralBenchCTNetClassifier(random_state=random_state, **kwargs)
    elif c in neuralbench_deep_models:
        from aeon_neuro.classification.experimental import NeuralBenchDeepClassifier

        _read_train_groups(kwargs)
        return NeuralBenchDeepClassifier(
            model=neuralbench_deep_models[c], random_state=random_state, **kwargs
        )
    elif c in ("differentialentropyclassifier", "de-lr", "de-svm"):
        from aeon_neuro.classification.feature_based import (
            DifferentialEntropyClassifier,
        )

        if c == "de-svm" and "classifier" not in kwargs:
            from sklearn.pipeline import make_pipeline
            from sklearn.preprocessing import StandardScaler
            from sklearn.svm import SVC

            kwargs["classifier"] = make_pipeline(
                StandardScaler(), SVC(probability=True, random_state=random_state)
            )
        return DifferentialEntropyClassifier(random_state=random_state, **kwargs)
    elif c in ("trcaclassifier", "trca"):
        from aeon_neuro.classification.distance_based import TRCAClassifier

        # deterministic: no random_state
        return TRCAClassifier(**kwargs)
    elif c == "fbtrca":
        from aeon_neuro.classification.distance_based import TRCAClassifier

        kwargs.setdefault("filter_bank", 5)
        return TRCAClassifier(**kwargs)
    elif c in ("trace", "traceclassifier"):
        from aeon_neuro.classification.distance_based import TRACE

        return TRACE(random_state=random_state, **kwargs)
    elif c in ("templateccaclassifier", "templatecca"):
        from aeon_neuro.classification.distance_based import TemplateCCAClassifier

        return TemplateCCAClassifier(**kwargs)
    elif c in moabb_pipelines:
        from aeon_neuro.classification.experimental import MOABBPipelineClassifier

        return MOABBPipelineClassifier(
            pipeline=moabb_pipelines[c],
            n_jobs=n_jobs,
            random_state=random_state,
            **kwargs,
        )
    else:
        raise ValueError(f"UNKNOWN CLASSIFIER: {c} in get_classifier_by_name")


def _read_train_groups(kwargs):
    """Turn a train_groups file path (or "none") from the command line into IDs.

    The file has one subject ID per training case, whitespace separated, in TRAIN
    file order. The IDs make the early stopping validation split subject-wise.
    """
    groups = kwargs.get("train_groups")
    if isinstance(groups, str):
        if groups.casefold() == "none":
            kwargs["train_groups"] = None
        else:
            from pathlib import Path

            path = Path(groups).expanduser()
            kwargs["train_groups"] = path.read_text(encoding="utf-8").split()


def _set_classifier_neuro_by_class_name(c, random_state, n_jobs, kwargs):
    """Create an aeon-neuro classifier exported by its classification package."""
    from importlib import import_module
    from inspect import isclass, signature

    from aeon.classification.base import BaseClassifier
    from aeon_neuro import classification

    # Some classifiers, e.g. CBraModClassifier, are only exported by a subpackage.
    candidates = {}
    for package in ("deep_learning", "distance_based", "feature_based"):
        try:
            candidates.update(
                vars(import_module(f"{classification.__name__}.{package}"))
            )
        except ImportError:
            pass
    candidates.update(vars(classification))
    for classifier in candidates.values():
        if (
            isclass(classifier)
            and classifier.__name__.casefold() == c
            and classifier.__module__.startswith("aeon_neuro.")
            and issubclass(classifier, BaseClassifier)
        ):
            parameters = signature(classifier).parameters
            init_kwargs = kwargs.copy()
            if "random_state" in parameters:
                init_kwargs.setdefault("random_state", random_state)
            if "n_jobs" in parameters:
                init_kwargs.setdefault("n_jobs", n_jobs)
            return classifier(**init_kwargs)

    raise ValueError(f"UNKNOWN CLASSIFIER: neuro:{c} in get_classifier_by_name")


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
    elif c == "tde-unequal":
        from tsml_eval._wip.unequal_length._tde7 import TemporalDictionaryEnsemble

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
    else:
        raise ValueError(f"UNKNOWN CLASSIFIER: {c} in get_classifier_by_name")


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
            n_jobs=n_jobs,
            **kwargs,
        )
    elif c == "grailclassifier" or c == "grail":
        from tsml.distance_based import GRAILClassifier

        return GRAILClassifier(**kwargs)
    elif c == "proximitytree" or c == "proximitytreeclassifier":
        from tsml_eval._wip.pf._pt import ProximityTree

        return ProximityTree(random_state=random_state, **kwargs)

    elif c == "proximityforest" or c == "pf":
        from tsml_eval._wip.pf._pf import ProximityForest

        return ProximityForest(random_state=random_state, n_jobs=n_jobs, **kwargs)
    elif c == "proximityforest2" or c == "pf2":
        from tsml_eval._wip.pf._pf2 import ProximityForest2

        return ProximityForest2(random_state=random_state, n_jobs=n_jobs, **kwargs)
    else:
        raise ValueError(f"UNKNOWN CLASSIFIER: {c} in get_classifier_by_name")


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
    elif c == "freshprince-unequal":
        from tsml_eval._wip.unequal_length._fresh_prince import FreshPRINCEClassifier

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
    else:
        raise ValueError(f"UNKNOWN CLASSIFIER: {c} in get_classifier_by_name")


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
    else:
        raise ValueError(f"UNKNOWN CLASSIFIER: {c} in get_classifier_by_name")


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
            time_limit_in_minutes=fit_contract,
            **kwargs,
        )
    elif c == "randomintervalspectralensembleclassifier" or c == "rise":
        from aeon.classification.interval_based import (
            RandomIntervalSpectralEnsembleClassifier,
        )

        return RandomIntervalSpectralEnsembleClassifier(
            random_state=random_state,
            n_jobs=n_jobs,
            time_limit_in_minutes=fit_contract,
            **kwargs,
        )
    elif c == "tsf-500":
        from aeon.classification.interval_based import TimeSeriesForestClassifier

        return TimeSeriesForestClassifier(
            n_estimators=500,
            random_state=random_state,
            n_jobs=n_jobs,
            time_limit_in_minutes=fit_contract,
            **kwargs,
        )
    elif c == "timeseriesforestclassifier" or c == "tsf":
        from aeon.classification.interval_based import TimeSeriesForestClassifier

        return TimeSeriesForestClassifier(
            random_state=random_state,
            n_jobs=n_jobs,
            time_limit_in_minutes=fit_contract,
            **kwargs,
        )
    elif c == "cif-500":
        from aeon.classification.interval_based import CanonicalIntervalForestClassifier

        return CanonicalIntervalForestClassifier(
            n_estimators=500,
            random_state=random_state,
            n_jobs=n_jobs,
            time_limit_in_minutes=fit_contract,
            **kwargs,
        )
    elif c == "canonicalintervalforestclassifier" or c == "cif":
        from aeon.classification.interval_based import CanonicalIntervalForestClassifier

        return CanonicalIntervalForestClassifier(
            random_state=random_state,
            n_jobs=n_jobs,
            time_limit_in_minutes=fit_contract,
            **kwargs,
        )
    elif c == "stsf-500":
        from aeon.classification.interval_based import SupervisedTimeSeriesForest

        return SupervisedTimeSeriesForest(
            n_estimators=500,
            random_state=random_state,
            n_jobs=n_jobs,
            time_limit_in_minutes=fit_contract,
            **kwargs,
        )
    elif c == "supervisedtimeseriesforest" or c == "stsf":
        from aeon.classification.interval_based import SupervisedTimeSeriesForest

        return SupervisedTimeSeriesForest(
            random_state=random_state,
            n_jobs=n_jobs,
            time_limit_in_minutes=fit_contract,
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
    elif c == "drcif-unequal":
        from tsml_eval._wip.unequal_length._drcif6 import DrCIFClassifier

        return DrCIFClassifier(
            n_estimators=500,
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

    elif c == "drcif-pipeline":
        import numpy as np
        from aeon.transformations.collection import (
            ARCoefficientTransformer,
            PeriodogramTransformer,
        )
        from aeon.transformations.collection.feature_based import Catch22
        from aeon.utils.numba.general import first_order_differences_3d
        from aeon.utils.numba.stats import (
            row_iqr,
            row_mean,
            row_median,
            row_numba_max,
            row_numba_min,
            row_ppv,
            row_slope,
            row_std,
        )
        from sklearn.ensemble import ExtraTreesClassifier
        from tsml.interval_based import RandomIntervalClassifier
        from tsml.transformations import FunctionTransformer

        def sqrt_times_15_plus_5_mv(X):
            return int(
                np.sqrt(X.shape[2]) * np.sqrt(X.shape[1]) * 15 + 5
            )  # pragma: no cover

        interval_features = [
            Catch22(outlier_norm=True, replace_nans=True),
            row_mean,
            row_std,
            row_slope,
            row_median,
            row_iqr,
            row_numba_min,
            row_numba_max,
            row_ppv,
        ]
        series_transformers = [
            None,
            FunctionTransformer(func=first_order_differences_3d, validate=False),
            PeriodogramTransformer(),
            ARCoefficientTransformer(replace_nan=True),
        ]

        return RandomIntervalClassifier(
            random_state=random_state,
            n_jobs=n_jobs,
            n_intervals=sqrt_times_15_plus_5_mv,
            features=interval_features,
            series_transformers=series_transformers,
            estimator=ExtraTreesClassifier(n_estimators=500, criterion="entropy"),
            **kwargs,
        )

    else:
        raise ValueError(f"UNKNOWN CLASSIFIER: {c} in get_classifier_by_name")


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
    else:
        raise ValueError(f"UNKNOWN CLASSIFIER: {c} in get_classifier_by_name")


def _set_classifier_shapelet_based(
    c, random_state, n_jobs, fit_contract, checkpoint, kwargs
):
    if c == "stc-2hour":
        from tsml_eval._wip.shapelets.early_abandon._stc3 import (
            ShapeletTransformClassifier,
        )

        return ShapeletTransformClassifier(
            transform_limit_in_minutes=120,
            random_state=random_state,
            n_jobs=n_jobs,
            **kwargs,
        )
    elif c == "shapelettransformclassifier" or c == "stc":
        from tsml_eval._wip.shapelets.early_abandon._stc3 import (
            ShapeletTransformClassifier,
        )

        return ShapeletTransformClassifier(
            random_state=random_state,
            n_jobs=n_jobs,
            time_limit_in_minutes=fit_contract,
            **kwargs,
        )
    elif c == "stc-aeon":
        from aeon.classification.shapelet_based import ShapeletTransformClassifier

        return ShapeletTransformClassifier(
            random_state=random_state,
            n_jobs=n_jobs,
            time_limit_in_minutes=fit_contract,
            **kwargs,
        )
    elif c == "stc-unequal":
        from tsml_eval._wip.unequal_length._stc7 import ShapeletTransformClassifier

        return ShapeletTransformClassifier(
            random_state=random_state,
            n_jobs=n_jobs,
            time_limit_in_minutes=fit_contract,
            **kwargs,
        )
    elif c == "rdstclassifier" or c == "rdst":
        from aeon.classification.shapelet_based import RDSTClassifier

        return RDSTClassifier(random_state=random_state, n_jobs=n_jobs, **kwargs)
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

        return SASTClassifier(random_state=random_state, n_jobs=n_jobs, **kwargs)
    elif c == "rsastclassifier" or c == "rsast":
        from aeon.classification.shapelet_based import RSASTClassifier

        return RSASTClassifier(random_state=random_state, n_jobs=n_jobs, **kwargs)
    else:
        raise ValueError(f"UNKNOWN CLASSIFIER: {c} in get_classifier_by_name")


def _set_classifier_vector(c, random_state, n_jobs, fit_contract, checkpoint, kwargs):
    if (
        c == "rotationforestclassifier" or c == "rotationforest" or c == "rotf"
    ) and checkpoint:
        # the same model, as an aeon classifier that can checkpoint and resume (the
        # sklearn route below is wrapped in a fresh clone, so it cannot); needs an
        # aeon with checkpointing, e.g. the ajb/hc2-components-checkpoint branch
        from tsml_eval.estimators.classification._checkpointed_rotation_forest import (
            CheckpointedRotationForestClassifier,
        )

        return CheckpointedRotationForestClassifier(
            random_state=random_state,
            n_jobs=n_jobs,
            time_limit_in_minutes=fit_contract,
            **kwargs,
        )
    elif c == "rotationforestclassifier" or c == "rotationforest" or c == "rotf":
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

        return LogisticRegression(random_state=random_state, **kwargs)
    elif c in ("svc", "svm", "supportvectorclassifier"):
        from sklearn.svm import SVC

        # probability=True so results files get probability estimates.
        return SVC(probability=True, random_state=random_state, **kwargs)
    elif c in ("randomforestclassifier", "randomforest", "rf"):
        from sklearn.ensemble import RandomForestClassifier

        return RandomForestClassifier(
            random_state=random_state, n_jobs=n_jobs, **kwargs
        )
    elif c == "prevalidatedridgeclassifier" or c == "prevalidatedridge":
        from aeon.classification.sklearn import PrevalidatedRidgeClassifier

        return PrevalidatedRidgeClassifier(**kwargs)
    else:
        raise ValueError(f"UNKNOWN CLASSIFIER: {c} in get_classifier_by_name")
