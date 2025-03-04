"""Test configuration."""

import os

if os.environ.get("CICD_RUNNING") == "1":
    import tsml_eval.testing._cicd_numba_caching  # noqa: F401
