"""FastDrCIF and FastDrCIF_D: length-gated and dilated random-interval DrCIF.

Both build on SharedDrCIF (random-interval DrCIF that reached QUANT accuracy
parity). FastDrCIF adds length-gated feature scaling (banding): each interval
computes only the catch22 features whose length threshold it clears. FastDrCIF_D
adds dilation on top: each random interval also draws a dilation scaled to its
length, expanding its window for a multi-scale view. All three variants
(SharedDrCIF, FastDrCIF, FastDrCIF_D) use random intervals, the same interval
count, and the constant-feature filter.
"""

__maintainer__ = ["TonyBagnall"]
__all__ = ["FastDrCIF", "FastDrCIF_D"]

from tsml_eval._wip.classification._shared_drcif import SharedDrCIF


class FastDrCIF(SharedDrCIF):
    """Random-interval DrCIF with length-gated feature scaling (banding).

    Same as SharedDrCIF (random intervals, constant-feature filter) but with
    ``banded=True`` by default: each interval computes only the catch22 features
    whose length threshold it clears, so short intervals skip the length-hungry
    features. See SharedDrCIF for full parameter and attribute documentation.
    """

    def __init__(
        self,
        features="drcif29",
        min_interval_length=3,
        max_interval_depth=6,
        max_interval_prop=0.5,
        banded=True,
        dilation=False,
        drop_constant=True,
        train_estimate=False,
        estimator=None,
        class_weight=None,
        random_state=None,
        n_jobs=1,
    ):
        super().__init__(
            features=features,
            interval_scheme="random",
            min_interval_length=min_interval_length,
            max_interval_depth=max_interval_depth,
            max_interval_prop=max_interval_prop,
            banded=banded,
            dilation=dilation,
            drop_constant=drop_constant,
            train_estimate=train_estimate,
            estimator=estimator,
            class_weight=class_weight,
            random_state=random_state,
            n_jobs=n_jobs,
        )


class FastDrCIF_D(FastDrCIF):
    """FastDrCIF with dilation added (length gates AND dilation).

    Same as FastDrCIF (random intervals, length gating, constant-feature filter)
    but with ``dilation=True`` by default: each random interval also draws a
    dilation scaled to its length (geometrically decaying towards d=1),
    expanding its window for a multi-scale view at no extra feature cost. See
    SharedDrCIF/FastDrCIF for full parameter documentation.
    """

    def __init__(
        self,
        features="drcif29",
        min_interval_length=3,
        max_interval_depth=6,
        max_interval_prop=0.5,
        banded=True,
        dilation=True,
        drop_constant=True,
        train_estimate=False,
        estimator=None,
        class_weight=None,
        random_state=None,
        n_jobs=1,
    ):
        super().__init__(
            features=features,
            min_interval_length=min_interval_length,
            max_interval_depth=max_interval_depth,
            max_interval_prop=max_interval_prop,
            banded=banded,
            dilation=dilation,
            drop_constant=drop_constant,
            train_estimate=train_estimate,
            estimator=estimator,
            class_weight=class_weight,
            random_state=random_state,
            n_jobs=n_jobs,
        )
