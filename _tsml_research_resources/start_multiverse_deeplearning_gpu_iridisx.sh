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
lock_file="${state_dir}/supervisor.lock"
mkdir -p "$state_dir"
export DISPLAY=
export MULTIVERSE_CLASSIFIER="$classifier"
export MULTIVERSE_STATE_DIR="$state_dir"
export MULTIVERSE_CLEAR_PENDING_ON_START=false
export MULTIVERSE_LOG_DIR="$log_dir"
cd "$repo_dir"
setsid nohup flock -n -E 75 "$lock_file" \
    env PYTHON="$python_executable" \
    bash "$script_dir/run_multiverse_controller.sh" "$config" \
    >> "$log_dir/launcher.out" 2>&1 < /dev/null &
launcher_pid=$!

# A failed non-blocking flock exits immediately with status 75. Give the detached
# launcher enough time to acquire the lock, then report duplicate starts clearly.
sleep 2
if ! kill -0 "$launcher_pid" 2>/dev/null; then
    set +e
    wait "$launcher_pid" 2>/dev/null
    launcher_status=$?
    set -e
    if [[ "$launcher_status" -eq 75 ]]; then
        echo "ERROR: a ${classifier} supervisor is already running." >&2
    else
        echo "ERROR: the ${classifier} supervisor exited during startup (status ${launcher_status})." >&2
        echo "Check ${log_dir}/launcher.out" >&2
    fi
    exit 1
fi

disown "$launcher_pid"
echo "$launcher_pid" > "$state_dir/launcher.pid"
echo "Started ${classifier} controller on i7_h200 (PID ${launcher_pid}); log: ${log_dir}/supervisor.log"
