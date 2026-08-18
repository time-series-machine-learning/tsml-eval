#!/bin/bash
# Activate the tsml-eval-gpu environment and run
# submit_hinception_bidmc_gpu_iridisx.py. Login nodes have no environment loaded by
# default, so a bare "python" call fails there without this.

set -euo pipefail

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)

module load conda/python3
conda_sh="$(dirname "$(dirname "$(command -v conda)")")/etc/profile.d/conda.sh"
if [[ ! -f "$conda_sh" ]]; then
    echo "ERROR: conda.sh not found at $conda_sh" >&2
    exit 1
fi
source "$conda_sh"
conda activate tsml-eval-gpu

python -u "${script_dir}/submit_hinception_bidmc_gpu_iridisx.py" "$@"
