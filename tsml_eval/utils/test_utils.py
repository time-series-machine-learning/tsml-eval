# -*- coding: utf-8 -*-

EXEMPT_ESTIMATOR_NAMES = ["ColumnEnsembleRegressor", "GridSearchCV"]


def _check_set_method(
    set_method, estimator_sub_list, estimator_dict, all_estimator_names
):
    for estimator_names in estimator_sub_list:
        estimator_names = (
            [estimator_names] if isinstance(estimator_names, str) else estimator_names
        )

        for estimator_alias in estimator_names:
            assert (
                estimator_alias not in all_estimator_names
            ), f"Estimator {estimator_alias} is duplicated"
            all_estimator_names.append(estimator_alias)

            try:
                e = set_method(estimator_alias)
            except ModuleNotFoundError:
                continue

            assert e is not None, f"Estimator {estimator_alias} not found"

            c_name = e.__class__.__name__
            if c_name == estimator_alias:
                estimator_dict[c_name] = True
            elif c_name not in estimator_dict:
                estimator_dict[c_name] = False
