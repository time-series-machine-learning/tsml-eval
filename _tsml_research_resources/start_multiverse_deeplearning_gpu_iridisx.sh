#!/bin/bash
# Run any aeon deep-learning classifier on all missing Multiverse-Core datasets.
set -euo pipefail
classifier=${1:-PatchMTSC}
script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
repo_dir=$(cd -- "${script_dir}/.." && pwd)
python_executable="/home/${USER}/.conda/envs/tsml-eval-gpu/bin/python"
config="${script_dir}/multiverse_core_resample0_convtran_gpu_iridisx_i7_h200.toml"
state_dir="/home/${USER}/Results/Multiverse/.controller-core-resample0-${classifier,,}-i7-h200"
log_dir="${state_dir}"
mkdir -p "$state_dir"
export DISPLAY=
export MULTIVERSE_CLASSIFIER="$classifier"
export MULTIVERSE_STATE_DIR="$state_dir"
export MULTIVERSE_CLEAR_PENDING_ON_START=false
export MULTIVERSE_LOG_DIR="$log_dir"
cd "$repo_dir"
setsid nohup env PYTHON="$python_executable" bash "$script_dir/run_multiverse_controller.sh" "$config" > "$log_dir/launcher.out" 2>&1 < /dev/null &
echo "Started ${classifier} controller on i7_h200; log: ${log_dir}/supervisor.log"
