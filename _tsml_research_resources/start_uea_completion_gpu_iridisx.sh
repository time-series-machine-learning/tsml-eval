#!/bin/bash
# Start both UEA-completion GPU controllers on IridisX's Early Access H200 partition.
#
# Completes the four UEA datasets absent from MultiverseCore: BasicMotions,
# FingerMovements, InsectWingbeat and SelfRegulationSCP2. Fifteen estimators already
# have the first three from earlier non-core passes and need only InsectWingbeat; the
# rest need all four. The controller skips anything with a testResample0.csv, so it
# works that out itself.
#
# Two controllers because gpu_check is per configuration: a torch check would pass while
# TensorFlow saw no GPU, so the PyTorch and Keras classifiers are queued separately.
#
# The CPU classifiers are handled on Hali by uea_completion_resample0_non_deep.toml,
# which this script does not touch.

set -euo pipefail

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
repo_dir=$(cd -- "${script_dir}/.." && pwd)
supervisor="${script_dir}/run_multiverse_controller.sh"

data_dir="/home/${USER}/Data/Multiverse"
results_dir="/home/${USER}/Results/Multiverse"
python_executable="/home/${USER}/.conda/envs/tsml-eval-gpu/bin/python"
required_branch="ajb/gpu"
dataset_list="${script_dir}/dataset_lists/MultivariateClassification4-UEANotInMultiverseCore.txt"

# config : state directory suffix
configs=(
    "uea_completion_resample0_deep_torch_gpu_iridisx_i7_h200.toml:deep-torch"
    "uea_completion_resample0_deep_keras_gpu_iridisx_i7_h200.toml:deep-keras"
)

for command_name in flock git pgrep pkill setsid squeue; do
    if ! command -v "$command_name" >/dev/null 2>&1; then
        echo "ERROR: required command is unavailable: ${command_name}" >&2
        exit 1
    fi
done
if [[ ! -x "$python_executable" ]]; then
    echo "ERROR: GPU-environment Python is not executable: ${python_executable}" >&2
    exit 1
fi
if [[ "$(git -C "$repo_dir" branch --show-current)" != "$required_branch" ]]; then
    echo "ERROR: UEA completion GPU jobs must run from ${required_branch}." >&2
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

# The four are not in MultiverseCore, so confirm the data is actually present before
# queueing anything. A missing dataset would otherwise fail once per estimator.
echo "Checking the four datasets are downloaded."
missing_data=0
while read -r dataset; do
    [[ -z "$dataset" ]] && continue
    if [[ ! -d "${data_dir}/${dataset}" ]]; then
        echo "  MISSING: ${data_dir}/${dataset}" >&2
        missing_data=1
    else
        echo "  ok: ${dataset}"
    fi
done < "$dataset_list"
if [[ "$missing_data" -ne 0 ]]; then
    echo "ERROR: download the missing datasets before starting." >&2
    exit 1
fi

cd "$repo_dir"
for entry in "${configs[@]}"; do
    config="${script_dir}/${entry%%:*}"
    state_dir="${results_dir}/.controller-uea-completion-resample0-${entry##*:}-i7-h200"
    config_name=$(basename "$config")
    label="${entry##*:}"

    if [[ ! -f "$config" ]]; then
        echo "ERROR: config not found: ${config}" >&2
        exit 1
    fi
    mkdir -p "$state_dir"

    echo
    echo "=== ${label} ==="
    "$python_executable" -u "${script_dir}/multiverse_controller.py" \
        --config "$config" --dry-run --no-email

    pkill -TERM -f "[r]un_multiverse_controller.sh.*${config_name}" || true
    pkill -TERM -f "[m]ultiverse_controller.py.*${config_name}" || true
    sleep 1

    # A killed supervisor orphans the sleep it waits in, and that orphan keeps the
    # inherited descriptor on supervisor.lock open, so flock -n would fail and the new
    # controller would exit silently while pgrep showed nothing running.
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
        echo "ERROR: ${label} controller did not remain running." >&2
        tail -20 "${state_dir}/launcher.out" >&2 || true
        if command -v fuser >/dev/null 2>&1 \
            && fuser -s "${state_dir}/supervisor.lock" 2>/dev/null; then
            echo "supervisor.lock is still held; see: fuser -v ${state_dir}/supervisor.lock" >&2
        fi
        exit 1
    fi
    echo "${label} controller started (launcher PID ${launcher_pid}), log ${state_dir}/supervisor.log"
done

echo
echo "Both UEA completion GPU controllers are running."
echo "InsectWingbeat is 30000 cases over 200 channels and is where these will spend"
echo "their time and memory; expect it to finish last, if at all, for the heavier ones."
echo
squeue -u "$USER" -p i7_h200
