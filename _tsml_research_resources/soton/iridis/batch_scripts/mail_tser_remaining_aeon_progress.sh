#!/bin/bash

set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

exec bash "${script_dir}/mail_tser_interval_progress.sh" \
    --profile remaining-aeon \
    "$@"
