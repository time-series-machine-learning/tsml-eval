#!/bin/bash
# Start both UEA-completion GPU controllers on IridisX's Early Access H200 partition.
#
# Completes the four UEA datasets absent from MultiverseCore: BasicMotions,
# FingerMovements, InsectWingbeat and SelfRegulationSCP2. Most estimators already have
# the first three from earlier non-core passes and need only InsectWingbeat; TS2Vec
# and DisjointCNN need all four. The controller skips anything with a
# testResample0.csv, so it works that out itself.
#
# TS2Vec has none of the four because its jobs died at fit on a bad import, fixed in
# a3ba18c; DisjointCNN because the port was only added to the Keras config afterwards.
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

# The four are not in MultiverseCore, so confirm the data is present AND loads the way
# the jobs will load it. Presence of the directory is not enough: InsectWingbeat ships
# unequal length, every deep classifier refuses unequal input, and the first attempt at
# it failed for all five torch estimators in under a minute. The archive also publishes
# an equal-length version, and classification_experiments.py passes load_equal_length,
# so aeon picks up <name>_eq_TRAIN.ts when it is sitting beside the original. This
# check loads each dataset exactly as the job will and requires a 3D array.
echo "Checking the four datasets load as equal length."
if ! "$python_executable" - "$dataset_list" "$data_dir" <<'PYTHON'
import sys
from aeon.datasets import load_classification

dataset_list, data_dir = sys.argv[1], sys.argv[2]
names = [line.strip() for line in open(dataset_list) if line.strip()]
bad = 0
for name in names:
    try:
        X, _ = load_classification(name, split="train", extract_path=data_dir,
                                   load_equal_length=True)
    except Exception as error:
        print(f"  FAILED to load {name}: {type(error).__name__}: {error}")
        bad += 1
        continue
    if hasattr(X, "ndim"):
        print(f"  ok: {name} {X.shape}")
    else:
        print(f"  UNEQUAL LENGTH: {name}, {len(X)} cases; every deep classifier will "
              f"refuse it. Copy the {name}_eq_TRAIN.ts/_eq_TEST.ts pair into "
              f"{data_dir}/{name}/ beside the originals.")
        bad += 1
raise SystemExit(1 if bad else 0)
PYTHON
then
    echo "ERROR: fix the data above before starting." >&2
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
