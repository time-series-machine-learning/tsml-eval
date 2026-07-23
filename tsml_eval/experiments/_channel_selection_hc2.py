"""Channel-selection classifier pipelines used by EEG experiments."""

__maintainer__ = ["TonyBagnall"]

from math import ceil

from aeon.classification.base import BaseClassifier
from sklearn.base import clone


class ChannelSelectionClassifierPipeline(BaseClassifier):
    """Fit a learned collection transform followed by a classifier.

    Unlike aeon's standard ``ClassifierPipeline``, this pipeline also supports
    resampling transforms whose ``fit_resample`` method returns aligned ``X`` and
    ``y``. This is required for ``CaseTimeReducer`` when it selects case
    subsampling. All other configured transforms retain ``y`` unchanged.

    Parameters
    ----------
    selector : str
        Channel-selection, channel-creation, or case/time-reduction method.
    classifier : classifier
        Classifier fitted after the learned transform.
    proportion : float, default=0.25
        Proportion of channels/components retained by methods with an explicit
        output-size parameter.
    random_state : int or None, default=None
        Random seed passed to stochastic transforms.
    n_jobs : int, default=1
        Number of jobs passed to transforms which support parallelism.
    """

    _tags = {
        "X_inner_type": "numpy3D",
        "capability:multivariate": True,
    }

    def __init__(
        self,
        selector,
        classifier,
        proportion=0.25,
        random_state=None,
        n_jobs=1,
    ):
        self.selector = selector
        self.classifier = classifier
        self.proportion = proportion
        self.random_state = random_state
        self.n_jobs = n_jobs
        super().__init__()

    def _fit(self, X, y):
        """Fit the transform and classifier, retaining aligned training labels."""
        self.transformer_ = _make_channel_transformer(
            self.selector,
            n_channels=X.shape[1],
            proportion=self.proportion,
            random_state=self.random_state,
            n_jobs=self.n_jobs,
        )

        if hasattr(self.transformer_, "fit_resample"):
            Xt, yt = self.transformer_.fit_resample(X, y)
        else:
            Xt = self.transformer_.fit_transform(X, y)
            yt = y

        self.classifier_ = clone(self.classifier)
        self.classifier_.fit(Xt, yt)
        return self

    def _predict(self, X):
        """Transform test cases without case subsampling, then predict."""
        return self.classifier_.predict(self.transformer_.transform(X))

    def _predict_proba(self, X):
        """Transform test cases without case subsampling, then predict probabilities."""
        return self.classifier_.predict_proba(self.transformer_.transform(X))


def _make_channel_transformer(
    selector,
    n_channels,
    proportion=0.25,
    random_state=None,
    n_jobs=1,
):
    """Construct a channel transform after the input channel count is known."""
    if not 0 < proportion <= 1:
        raise ValueError("proportion must be in the range (0, 1].")

    selector_key = selector.casefold()
    n_components = max(1, ceil(proportion * n_channels))

    if selector_key == "ecs":
        from aeon.transformations.collection.channel_selection import ElbowClassSum

        return ElbowClassSum()
    if selector_key == "ecp":
        from aeon.transformations.collection.channel_selection import (
            ElbowClassPairwise,
        )

        return ElbowClassPairwise()
    if selector_key == "random":
        from aeon.transformations.collection.channel_selection import (
            RandomChannelSelector,
        )

        return RandomChannelSelector(p=proportion, random_state=random_state)
    if selector_key == "tselect":
        from aeon.transformations.collection.channel_selection import TSelect

        return TSelect(random_state=random_state)
    if selector_key == "csp":
        from aeon_neuro.transformations.collection.channel_creation import (
            CommonSpacialPatterns,
        )

        return CommonSpacialPatterns(
            n_components=n_components,
            log=None,
            transform_into="csp_space",
            random_state=random_state,
        )
    if selector_key == "riemannian":
        from aeon_neuro.transformations.collection.channel_selection import Riemannian

        return Riemannian(
            proportion=proportion,
            regularization=1e-6,
            n_jobs=n_jobs,
        )
    if selector_key == "detachrocket":
        from aeon_neuro.transformations.collection.channel_selection import (
            DetachRocketChannelSelector,
        )

        return DetachRocketChannelSelector(
            proportion=proportion,
            n_kernels=2000,
            n_jobs=n_jobs,
            random_state=random_state,
        )
    if selector_key == "casetimereducer":
        from aeon_neuro.transformations.collection.channel_selection import (
            CaseTimeReducer,
        )

        return CaseTimeReducer(
            strategy="auto",
            random_state=random_state,
            n_jobs=n_jobs,
        )
    if selector_key == "cleverrank":
        from aeon_neuro.transformations.collection.channel_selection import CLeVerRank

        return CLeVerRank(n_channels=n_components)
    if selector_key == "clevercluster":
        from aeon_neuro.transformations.collection.channel_selection import (
            CLeVerCluster,
        )

        return CLeVerCluster(
            n_channels=n_components,
            random_state=random_state,
        )
    if selector_key == "cleverhybrid":
        from aeon_neuro.transformations.collection.channel_selection import (
            CLeVerHybrid,
        )

        return CLeVerHybrid(
            n_channels=n_components,
            random_state=random_state,
        )

    raise ValueError(f"Unknown channel selector: {selector}")
