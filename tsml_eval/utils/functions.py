"""Miscellaneous functions for tsml_eval."""

__all__ = [
    "str_in_nested_list",
    "pair_list_to_dict",
]


def str_in_nested_list(nested_list, item):
    """Find an item in a nested list."""
    if item in (s.casefold() for s in nested_list if isinstance(s, str)):
        return True
    else:
        return any(
            str_in_nested_list(nl, item) for nl in nested_list if isinstance(nl, list)
        )


def pair_list_to_dict(pl):
    """Convert a 2d list of pairs to a dict.

    Each list item must be a tuple or list of length 2. The first item in each pair
    is used as the key, the second as the value.

    If ls is None, returns an empty dict.
    """
    return {} if pl is None else {k: v for k, v in pl}
