"""QuantDrCIF classifier.

DrCIF with the per-tree attribute pool extended from 29 to 35 by six QUANT-style
quantile features: Q10, Q25, Q75, Q90, and mean-centred Q25/Q75. Q0/Q50/Q100 are
excluded as they duplicate the min/median/max summary stats already in the pool.

Experimental arms are configured via att_subsample_size alone:
arm A (cost-neutral) uses the default 10, arm B (enriched, ~20% more extraction)
uses 12 to preserve the catch22 sampling density.
"""

__maintainer__ = ["TonyBagnall"]
__all__ = ["QuantDrCIF"]

from tsml_eval._wip.classification._new_drcif import NewDrCIF
from tsml_eval._wip.classification._quantile_stats import (
    row_quantile_10,
    row_quantile_25,
    row_quantile_25_centred,
    row_quantile_75,
    row_quantile_75_centred,
    row_quantile_90,
)


class QuantDrCIF(NewDrCIF):
    """DrCIF with six quantile features added to the attribute pool.

    Identical to DrCIF in every other respect; takes the same parameters.
    See NewDrCIF/DrCIFClassifier for the parameter and attribute documentation.
    """

    def __init__(
        self,
        base_estimator=None,
        n_estimators=200,
        n_intervals=(4, "sqrt-div"),
        min_interval_length=3,
        max_interval_length=0.5,
        att_subsample_size=10,
        time_limit_in_minutes=None,
        contract_max_n_estimators=500,
        use_pycatch22=False,
        random_state=None,
        n_jobs=1,
        parallel_backend=None,
    ):
        super().__init__(
            base_estimator=base_estimator,
            n_estimators=n_estimators,
            n_intervals=n_intervals,
            min_interval_length=min_interval_length,
            max_interval_length=max_interval_length,
            att_subsample_size=att_subsample_size,
            time_limit_in_minutes=time_limit_in_minutes,
            contract_max_n_estimators=contract_max_n_estimators,
            use_pycatch22=use_pycatch22,
            random_state=random_state,
            n_jobs=n_jobs,
            parallel_backend=parallel_backend,
        )

        self.interval_features = self.interval_features + [
            row_quantile_10,
            row_quantile_25,
            row_quantile_75,
            row_quantile_90,
            row_quantile_25_centred,
            row_quantile_75_centred,
        ]
