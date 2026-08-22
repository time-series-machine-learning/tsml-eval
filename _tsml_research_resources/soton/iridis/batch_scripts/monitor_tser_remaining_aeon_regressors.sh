#!/bin/bash

set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

exec bash "${script_dir}/monitor_tser_interval_regressors.sh" \
    --profile remaining-aeon \
    "$@"
