"""Piecewise Linear Approximation."""

__all__ = [
    "BasePLA",
    "SlidingWindow",
    "TopDown",
    "BottomUp",
    "SWAB",
]
from base import BasePLA
from _sw import SlidingWindow
from _td import TopDown
from _bu import BottomUp
from _swab import SWAB