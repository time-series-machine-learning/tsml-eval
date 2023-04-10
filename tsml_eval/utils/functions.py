# -*- coding: utf-8 -*-
"""Miscellaneous functions for tsml_eval."""


def str_in_nested_list(l, i):
    """Find an item in a nested list."""
    if i in (s.casefold() for s in l if isinstance(s, str)):
        return True
    else:
        return any(str_in_nested_list(nl, i) for nl in l if isinstance(nl, list))
