"""FastDrCIF: SharedDrCIF with DrCIF's random interval model.

Identical to SharedDrCIF except intervals are drawn once using DrCIF's random
interval generation rule (50/50 start/end anchor, length in
[min_interval_length, max_interval_prop * m]) instead of the dyadic
power-of-two grid. The interval count per representation matches the dyadic
grid, so SharedDrCIF vs FastDrCIF isolates the interval scheme (fixed dyadic
positions vs DrCIF-style random positions/lengths) with feature dimensionality
held constant. FastDrCIF is the random-interval variant that reached accuracy
parity with QUANT.
"""

__maintainer__ = ["TonyBagnall"]
__all__ = ["FastDrCIF"]

from tsml_eval._wip.classification._shared_drcif import SharedDrCIF


class FastDrCIF(SharedDrCIF):
    """SharedDrCIF using DrCIF's random interval model instead of a dyadic grid.

    See SharedDrCIF for the full parameter and attribute documentation. The
    only behavioural difference is ``interval_scheme="random"`` by default,
    which draws intervals with DrCIF's generation rule seeded by
    ``random_state`` (so different resamples get different interval sets, as in
    DrCIF).
    """

    def __init__(
        self,
        features="drcif29",
        min_interval_length=3,
        max_interval_depth=6,
        max_interval_prop=0.5,
        banded=False,
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
            train_estimate=train_estimate,
            estimator=estimator,
            class_weight=class_weight,
            random_state=random_state,
            n_jobs=n_jobs,
        )
