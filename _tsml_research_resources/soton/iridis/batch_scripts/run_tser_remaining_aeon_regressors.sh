#!/bin/bash

set -euo pipefail

# Run the non-deep aeon regressors not covered by the interval-only TSER pass.
# The shared driver keeps the same four-node task-farm, restart, active-job and
# memory-escalation behaviour as run_tser_interval_regressors.sh.

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

exec bash "${script_dir}/run_tser_interval_regressors.sh" \
    --profile remaining-aeon \
    "$@"
