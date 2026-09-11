#!/bin/bash
# Read-only: same configuration and completion rules as the runner.
set -euo pipefail
script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
python_executable="${UCR_REFERENCE_PYTHON:-/home/ajb2u23/.conda/envs/tsml-eval/bin/python}"
exec "${python_executable}" -u "${script_dir}/controller.py" monitor "$@"
