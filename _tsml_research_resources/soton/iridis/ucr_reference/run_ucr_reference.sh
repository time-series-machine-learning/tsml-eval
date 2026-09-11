#!/bin/bash
# Configure ucr_reference.json, then run --check and --dry-run before launch.
set -euo pipefail
script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
python_executable="${UCR_REFERENCE_PYTHON:-/home/ajb2u23/.conda/envs/tsml-eval/bin/python}"
exec "${python_executable}" -u "${script_dir}/controller.py" run "$@"
