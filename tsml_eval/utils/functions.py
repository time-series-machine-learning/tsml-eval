# -*- coding: utf-8 -*-
"""Miscellaneous functions for tsml_eval."""

__all__ = ["str_in_nested_list"]


def str_in_nested_list(nested_list, item):
    """Find an item in a nested list."""
    if item in (s.casefold() for s in nested_list if isinstance(s, str)):
        return True
    else:
        return any(
            str_in_nested_list(nl, item) for nl in nested_list if isinstance(nl, list)
        )
