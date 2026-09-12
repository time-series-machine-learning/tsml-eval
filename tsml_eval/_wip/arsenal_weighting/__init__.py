"""Arsenal with a corrected member weighting."""

__all__ = ["FixedWeightArsenal", "CVWeightArsenal", "EqualWeightArsenal"]

from tsml_eval._wip.arsenal_weighting._arsenal_fixed import (
    CVWeightArsenal,
    EqualWeightArsenal,
    FixedWeightArsenal,
)
