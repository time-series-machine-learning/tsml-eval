#!/bin/bash
# Show what the Multiverse reference run has done and what is left.
set -euo pipefail
here=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
python="${MULTIVERSE_REFERENCE_PYTHON:-/home/${USER}/.conda/envs/tsml-eval/bin/python}"
exec "$python" "${here}/../ucr_reference/controller.py" \
    --config "${here}/multiverse_reference.json" --watch "${1:-0}" --details
