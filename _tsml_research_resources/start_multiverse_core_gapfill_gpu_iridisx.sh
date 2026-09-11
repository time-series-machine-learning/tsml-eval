#!/bin/bash
# Start the deep-learner gap-fill controller on IridisX's Early Access H200 queue.
#
# Five experiments, each closing a dataset or bringing one within one job of it:
#
#   ConvTran   Alzheimers, PhotoStimulation   shared a GPU; now one job per node
#   TS2Vec     Locust2022, Tiselac, USCActivity           60 hour probe timeouts
#
# ConvTran's three complete their datasets outright, taking the scored set from
# 56 to 59 of 64.
#
#   more than the 139.8 GiB card holds, so it needs a batch size the paper does not
#   specify and that decision has not been made.
#
# One controller now, not two. LiteTIME is withheld from the tables, so the Keras
# half no longer closes anything, and AustraliaRainfall_disc is deferred from the
# collection, so it is out of the dataset list.
#
# TS2Vec will repeat its timeouts without the current checkout: it needs both the
# vendored-package import fix and the probe fix, so this refuses to run against a
# stale tree.

set -euo pipefail

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
repo_dir=$(cd -- "${script_dir}/.." && pwd)
supervisor="${script_dir}/run_multiverse_controller.sh"

data_dir="/home/${USER}/Data/Multiverse"
results_dir="/home/${USER}/Results/Multiverse"
python_executable="/home/${USER}/.conda/envs/tsml-eval-gpu/bin/python"
required_branch="ajb/gpu"
dataset_list="${script_dir}/dataset_lists/MultivariateClassification5-DeepGaps.txt"

# config : state directory suffix
configs=(
    "multiverse_core_gapfill_deep_torch_gpu_iridisx_i7_h200.toml:torch"
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
if [[ ! -f "$dataset_list" ]]; then
    echo "ERROR: dataset list not found: ${dataset_list}" >&2
    exit 1
fi
if [[ "$(git -C "$repo_dir" branch --show-current)" != "$required_branch" ]]; then
    echo "ERROR: gap-fill GPU jobs must run from ${required_branch}." >&2
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

# Show what is actually missing, rather than trusting the list. A count of zero for
# a classifier means its gaps closed since this list was written and it will queue
# nothing, which is fine; a count of nine means its results directory has gone.
echo "Gaps the controllers will submit:"
pending_total=0
for classifier in ConvTran TS2Vec; do
    predictions_dir="${results_dir}/DeepLearning/${classifier}/Predictions"
    pending=""
    while read -r dataset; do
        [[ -z "$dataset" ]] && continue
        result="${predictions_dir}/${dataset}/testResample0.csv"
        if [[ ! -s "$result" ]]; then
            pending+=" ${dataset}"
            pending_total=$((pending_total + 1))
        fi
    done < "$dataset_list"
    echo "  ${classifier}:${pending:- none}"
done
echo "  ${pending_total} job(s) in total"
if [[ "$pending_total" -eq 0 ]]; then
    echo "Nothing to do; every gap in the list is already filled."
    exit 0
fi

cd "$repo_dir"
for entry in "${configs[@]}"; do
    config="${script_dir}/${entry%%:*}"
    label="${entry##*:}"
    state_dir="${results_dir}/.controller-core-gapfill-deep-${label}-i7-h200"
    config_name=$(basename "$config")

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
    # inherited descriptor on supervisor.lock open, so flock -n would fail and the
    # new controller would exit silently while pgrep showed nothing running.
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
echo "The gap-fill controller is running."
echo "ConvTran's three failed with CUDA out of memory on the A100 partition; the H200"
echo "NVL carries 141 GB against 40 or 80 there, which is the reason to expect these"
echo "to pass rather than any change to memory_mb_levels, which governs host RAM."
echo
squeue -u "$USER" -p i7_h200
