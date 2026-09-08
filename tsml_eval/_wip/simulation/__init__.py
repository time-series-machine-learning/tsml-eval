"""Work-in-progress simulators for time series classification.

Each simulator generates a labelled collection for which one *family* of TSC
algorithms should be (close to) optimal, so that classifier behaviour can be
studied against a known generative truth. Draft ports/redesigns of the Java
simulators in ``tsml-java`` (``statistics.simulators``), intended for aeon.

``_protocol`` is the executable form of the explanatory simulation protocol in
the interval-based review, and is authoritative where it and the earlier
``simulate_interval_shape_data`` simulator disagree.
"""

__all__ = [
    "simulate_interval_shape_data",
    "SHAPES",
    "order_templates",
    "level_offsets",
    "simulate_protocol_problem",
    "protocol_conditions",
    "condition_id",
    "support_length_amplitude",
    "kl_matched_variance_ratio",
    "MECHANISMS",
]

from tsml_eval._wip.simulation._interval import (
    SHAPES,
    level_offsets,
    order_templates,
    simulate_interval_shape_data,
)
from tsml_eval._wip.simulation._protocol import (
    MECHANISMS,
    condition_id,
    kl_matched_variance_ratio,
    protocol_conditions,
    simulate_protocol_problem,
    support_length_amplitude,
)
