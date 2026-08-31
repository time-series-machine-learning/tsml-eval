"""TDE development variant which forces bigrams for multivariate data."""

import numpy as np
from aeon.classification.dictionary_based import TemporalDictionaryEnsemble

__all__ = ["TDE_Dev2"]


class TDE_Dev2(TemporalDictionaryEnsemble):
    """Temporal Dictionary Ensemble with bigrams enabled by default.

    This is an independent ablation of standard TDE. All standard TDE defaults are
    retained except ``bigrams=True``. In particular, it retains the standard channel
    filtering defaults ``dim_threshold=0.85`` and ``max_dims=20``.
    """

    def __init__(
        self,
        n_parameter_samples=250,
        max_ensemble_size=50,
        max_win_len_prop=1,
        min_window=10,
        randomly_selected_params=50,
        bigrams=True,
        dim_threshold=0.85,
        max_dims=20,
        time_limit_in_minutes=0.0,
        contract_max_n_parameter_samples=np.inf,
        typed_dict="deprecated",
        train_estimate_method="loocv",
        n_jobs=1,
        random_state=None,
        verbose=0,
    ):
        super().__init__(
            n_parameter_samples=n_parameter_samples,
            max_ensemble_size=max_ensemble_size,
            max_win_len_prop=max_win_len_prop,
            min_window=min_window,
            randomly_selected_params=randomly_selected_params,
            bigrams=bigrams,
            dim_threshold=dim_threshold,
            max_dims=max_dims,
            time_limit_in_minutes=time_limit_in_minutes,
            contract_max_n_parameter_samples=contract_max_n_parameter_samples,
            typed_dict=typed_dict,
            train_estimate_method=train_estimate_method,
            n_jobs=n_jobs,
            random_state=random_state,
            verbose=verbose,
        )
