#!/bin/bash
# Fill the Multiverse-core resample-0 results that the D-drive reference lacks.
#
# Companion to ucr_reference, sharing its controller: the same manifest model,
# the same task farms, the same monitor. Only the configuration differs.
set -euo pipefail
here=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
controller="${here}/../ucr_reference/controller.py"
python="${MULTIVERSE_REFERENCE_PYTHON:-/home/${USER}/.conda/envs/tsml-eval/bin/python}"
exec "$python" "$controller" \
    --config "${here}/multiverse_reference.json" \
    --device "${MULTIVERSE_REFERENCE_DEVICE:-cpu}" \
    "$@"
