"""get data transformer function."""

__maintainer__ = ["MatthewMiddlehurst"]

from aeon.transformations.collection import Normalizer

from tsml_eval.utils.functions import str_in_nested_list

transformers = [
    ["normalizer", "normaliser"],
    "padder",
]


def get_data_transform_by_name(
    transformer_names,
    row_normalise=False,
    random_state=None,
    n_jobs=1,
):
    """ """
    if transformer_names is None and not row_normalise:
        return None

    t_list = []
    if row_normalise:
        t_list.append(Normalizer())

    if transformer_names is not None:
        if not isinstance(transformer_names, list):
            transformer_names = [transformer_names]

        for transformer_name in transformer_names:
            t = transformer_name.casefold()

            if str_in_nested_list(transformers, t):
                t_list.append(_set_transformer(t, random_state, n_jobs))
            else:
                raise ValueError(
                    f"UNKNOWN TRANSFORMER: {t} in get_data_transform_by_name"
                )

    return t_list if len(t_list) > 1 else t_list[0]


def _set_transformer(t, random_state, n_jobs):
    if t == "normalizer" or t == "normaliser":
        return Normalizer()
    elif t == "padder":
        from aeon.transformations.collection import Padder

        return Padder()
