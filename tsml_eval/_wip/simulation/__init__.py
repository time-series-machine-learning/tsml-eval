"""Work-in-progress simulators for time series classification.

Each simulator generates a labelled collection for which one *family* of TSC
algorithms should be (close to) optimal, so that classifier behaviour can be
studied against a known generative truth. Draft ports/redesigns of the Java
simulators in ``tsml-java`` (``statistics.simulators``), intended for aeon.
"""

__all__ = ["simulate_interval_shape_data", "SHAPES"]

from tsml_eval._wip.simulation._interval import SHAPES, simulate_interval_shape_data
