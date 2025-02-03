"""get data transformer function."""

__maintainer__ = ["MatthewMiddlehurst"]

from aeon.transformations.collection import Normalizer

from tsml_eval.utils.functions import str_in_nested_list

transformers = [
    ["normalizer", "normaliser"],
    ["padder", "zero-padder"],
    "low-noise-padder",
]


def get_data_transform_by_name(
    transformer_names,
    row_normalise=False,
    random_state=None,
    n_jobs=1,
):
    """Return a transformers matching a given input name(s).

    Parameters
    ----------
    transformer_names : str or list of str
        String or list of strings indicating the transformer(s) to be returned.
    row_normalise : bool, default=False
        Adds a Normalizer to the front of the transformer list.
    random_state : int, RandomState instance or None, default=None
        Random seed or RandomState object to be used in the classifier if available.
    n_jobs: int, default=1
        The number of jobs to run in parallel for both classifier ``fit`` and
        ``predict`` if available. `-1` means using all processors.

    Return
    ------
    transformers : A transformer or list of transformers.
        The transformer(s) matching the input transformer name(s). Returns a list if
        more than one transformer is requested.
    """
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
    elif t == "padder" or t == "zero-padder":
        from aeon.transformations.collection import Padder

        return Padder()
    elif t == "low-noise-padder":
        from tsml_eval._wip.unequal_length._pad import Padder

        return Padder(add_noise=0.1, random_state=random_state)
