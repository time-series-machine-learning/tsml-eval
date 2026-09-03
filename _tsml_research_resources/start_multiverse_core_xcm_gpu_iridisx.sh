#!/bin/bash
# Start the XCM resample-0 controller on IridisX's Early Access H200 partition.
#
# Simpler than the ConvTran starter in two ways. There is one controller, not two,
# because XCM is only queued on i7_h200. And there is no feasibility selector: the
# ConvTran one exists to keep quadratic attention work inside the GPU, which does not
# apply to a convolutional network, so XCM is offered the whole MultiverseCore list and
# the controller's memory escalation handles the heavy problems.

set -euo pipefail

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
repo_dir=$(cd -- "${script_dir}/.." && pwd)
supervisor="${script_dir}/run_multiverse_controller.sh"
config="${script_dir}/multiverse_core_resample0_xcm_gpu_iridisx_i7_h200.toml"

data_dir="/home/${USER}/Data/Multiverse"
results_dir="/home/${USER}/Results/Multiverse"
state_dir="${results_dir}/.controller-core-resample0-xcm-i7-h200"
python_executable="/home/${USER}/.conda/envs/tsml-eval-gpu/bin/python"
required_branch="ajb/gpu"

for command_name in flock git pgrep pkill setsid squeue; do
    if ! command -v "$command_name" >/dev/null 2>&1; then
        echo "ERROR: required command is unavailable: ${command_name}" >&2
        exit 1
    fi
done
for required_file in "$supervisor" "$config" "$python_executable"; do
    if [[ ! -f "$required_file" ]]; then
        echo "ERROR: required file not found: ${required_file}" >&2
        exit 1
    fi
done
if [[ ! -x "$python_executable" ]]; then
    echo "ERROR: GPU-environment Python is not executable: ${python_executable}" >&2
    exit 1
fi
if [[ "$(git -C "$repo_dir" branch --show-current)" != "$required_branch" ]]; then
    echo "ERROR: XCM GPU jobs must run from ${required_branch}." >&2
    exit 1
fi
if [[ -n "$(git -C "$repo_dir" status --porcelain --untracked-files=normal)" ]]; then
    echo "ERROR: commit or discard repository changes before submission." >&2
    echo "The controller pins HEAD, so an uncommitted experiment is not reproducible." >&2
    exit 1
fi
if [[ ! -d "$data_dir" ]]; then
    echo "ERROR: Multiverse data directory not found: ${data_dir}" >&2
    exit 1
fi

# XCM is a Keras port, so confirm this environment's TensorFlow can see a GPU builder
# before queueing anything. The controller's own preflight runs inside the job, which
# is too late to save a whole submission round.
"$python_executable" -c "import tensorflow as tf; print('TensorFlow', tf.__version__)"
"$python_executable" -c "from tsml_eval.experiments import get_classifier_by_name;
print('XCM ->', type(get_classifier_by_name('XCM', random_state=0)).__name__)"

mkdir -p "$state_dir"

cd "$repo_dir"
echo "Dry-running the i7_h200 XCM controller."
"$python_executable" -u "${script_dir}/multiverse_controller.py" \
    --config "$config" --dry-run --no-email

# Replace only a previous supervisor for this exact configuration. Running Slurm jobs,
# including any other classifier's, are left alone.
config_name=$(basename "$config")
pkill -TERM -f "[r]un_multiverse_controller.sh.*${config_name}" || true
pkill -TERM -f "[m]ultiverse_controller.py.*${config_name}" || true
sleep 1

# Killing a supervisor orphans the "sleep" it waits in between cycles, and that
# orphan keeps the inherited descriptor on supervisor.lock open. A flock lock
# lives as long as any descriptor on the file does, so the next launch would
# fail to acquire and exit silently, while pgrep showed no controller running.
# Release the lock explicitly before launching.
if [[ -e "${state_dir}/supervisor.lock" ]] && command -v fuser >/dev/null 2>&1; then
    if fuser -s "${state_dir}/supervisor.lock" 2>/dev/null; then
        echo "Releasing supervisor.lock held by an orphaned process."
        fuser -k -TERM "${state_dir}/supervisor.lock" >/dev/null 2>&1 || true
        sleep 1
    fi
fi

setsid nohup flock -n "${state_dir}/supervisor.lock" \
    env PYTHON="$python_executable" \
        MULTIVERSE_CLEAR_PENDING_ON_START=false \
        MULTIVERSE_LOG_DIR="$state_dir" \
    bash "$supervisor" "$config" \
    > "${state_dir}/launcher.out" 2>&1 < /dev/null &
launcher_pid=$!
disown "$launcher_pid"
echo "$launcher_pid" > "${state_dir}/launcher.pid"
sleep 2
if ! pgrep -f "[r]un_multiverse_controller.sh.*${config_name}" >/dev/null; then
    echo "ERROR: XCM controller did not remain running." >&2
    echo "Check ${state_dir}/launcher.out:" >&2
    tail -20 "${state_dir}/launcher.out" >&2 || true
    if command -v fuser >/dev/null 2>&1 \n        && fuser -s "${state_dir}/supervisor.lock" 2>/dev/null; then
        echo "supervisor.lock is still held; see: fuser -v ${state_dir}/supervisor.lock" >&2
    fi
    exit 1
fi
echo "XCM controller started (launcher PID ${launcher_pid})."

echo
echo "XCM resample-0 controller is running."
echo "log: ${state_dir}/supervisor.log"
echo
squeue -u "$USER" -p i7_h200
