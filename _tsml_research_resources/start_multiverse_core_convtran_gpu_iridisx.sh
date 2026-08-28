#!/bin/bash
# Select a conservative ConvTran subset of MultiverseCore, split it between the two
# IridisX H200 partitions, dry-run both controllers, and start them detached.

set -euo pipefail

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
repo_dir=$(cd -- "${script_dir}/.." && pwd)
core_list_name="MultivariateClassification66-MultiverseMini.txt"
source_list="${script_dir}/dataset_lists/${core_list_name}"
selector="${script_dir}/select_convtran_datasets.py"
supervisor="${script_dir}/run_multiverse_controller.sh"
config_prefix="${script_dir}/multiverse_core_resample0_convtran_gpu_iridisx"
config_quad="${config_prefix}_quad_h200.toml"
config_i7="${config_prefix}_i7_h200.toml"

data_dir="/home/${USER}/Data/Multiverse"
list_dir="/home/${USER}/DataSetLists"
feasible_list="${list_dir}/MultiverseCore-ConvTran-Feasible.txt"
rejected_list="${list_dir}/MultiverseCore-ConvTran-Rejected.tsv"
unavailable_list="${list_dir}/MultiverseCore-ConvTran-Unavailable.txt"
quad_list="${list_dir}/MultiverseCore-ConvTran-QuadH200.txt"
i7_list="${list_dir}/MultiverseCore-ConvTran-I7H200.txt"
results_dir="/home/${USER}/Results/Multiverse"
state_quad="${results_dir}/.controller-core-resample0-convtran-quad-h200"
state_i7="${results_dir}/.controller-core-resample0-convtran-i7-h200"
python_executable="/home/${USER}/.conda/envs/tsml-eval-gpu/bin/python"
required_branch="ajb/gpu"

# This is a deliberately conservative first pass. Increase these only after reviewing
# timing and GPU-memory use from the completed jobs.
max_train_cases=${CONVTRAN_MAX_TRAIN_CASES:-10000}
max_timepoints=${CONVTRAN_MAX_TIMEPOINTS:-2000}
max_attention_work=${CONVTRAN_MAX_ATTENTION_WORK:-1000000000}

for command_name in flock git pgrep pkill setsid squeue; do
    if ! command -v "$command_name" >/dev/null 2>&1; then
        echo "ERROR: required command is unavailable: ${command_name}" >&2
        exit 1
    fi
done
for required_file in "$source_list" "$selector" "$supervisor" "$config_quad" \
    "$config_i7" "$python_executable"; do
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
    echo "ERROR: ConvTran GPU jobs must run from ${required_branch}." >&2
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

mkdir -p "$list_dir" "$state_quad" "$state_i7"
"$python_executable" "$selector" \
    --source-list "$source_list" \
    --data-dir "$data_dir" \
    --output "$feasible_list" \
    --rejected-output "$rejected_list" \
    --unavailable-output "$unavailable_list" \
    --max-train-cases "$max_train_cases" \
    --max-timepoints "$max_timepoints" \
    --max-attention-work "$max_attention_work"

feasible_count=$(wc -l < "$feasible_list")
if ((feasible_count < 2)); then
    echo "ERROR: fewer than two feasible datasets; refusing dual-queue submission." >&2
    exit 1
fi

# Alternate the workload-sorted list so both partitions receive a size mix and the
# two controllers can never race to submit the same classifier/dataset task.
awk 'NR % 2 == 1' "$feasible_list" > "$quad_list"
awk 'NR % 2 == 0' "$feasible_list" > "$i7_list"
echo "quad_h200: $(wc -l < "$quad_list") datasets -> ${quad_list}"
echo "i7_h200:   $(wc -l < "$i7_list") datasets -> ${i7_list}"
echo "Rejected:  ${rejected_list}"
echo "Missing:   ${unavailable_list}"

cd "$repo_dir"
echo "Dry-running quad_h200 controller."
"$python_executable" -u "$selector" --help >/dev/null
"$python_executable" -u "${script_dir}/multiverse_controller.py" \
    --config "$config_quad" --dry-run --no-email
echo "Dry-running i7_h200 controller."
"$python_executable" -u "${script_dir}/multiverse_controller.py" \
    --config "$config_i7" --dry-run --no-email

# Replace only previous supervisors for these exact configurations. Existing running
# Slurm jobs are left alone and are reconciled by the new controller cycles.
for config in "$config_quad" "$config_i7"; do
    config_name=$(basename "$config")
    pkill -TERM -f "[r]un_multiverse_controller.sh.*${config_name}" || true
    pkill -TERM -f "[m]ultiverse_controller.py.*${config_name}" || true
done
sleep 1

start_controller() {
    local config=$1
    local state_dir=$2
    local label=$3
    local config_name
    config_name=$(basename "$config")

    setsid nohup flock -n "${state_dir}/supervisor.lock" \
        env PYTHON="$python_executable" \
            MULTIVERSE_CLEAR_PENDING_ON_START=false \
            MULTIVERSE_LOG_DIR="$state_dir" \
        bash "$supervisor" "$config" \
        > "${state_dir}/launcher.out" 2>&1 < /dev/null &
    local launcher_pid=$!
    disown "$launcher_pid"
    echo "$launcher_pid" > "${state_dir}/launcher.pid"
    sleep 2
    if ! pgrep -f "[r]un_multiverse_controller.sh.*${config_name}" >/dev/null; then
        echo "ERROR: ${label} controller did not remain running." >&2
        echo "Check ${state_dir}/launcher.out" >&2
        exit 1
    fi
    echo "${label} controller started (launcher PID ${launcher_pid})."
}

start_controller "$config_quad" "$state_quad" "quad_h200"
start_controller "$config_i7" "$state_i7" "i7_h200"

echo
echo "ConvTran resample-0 controllers are running."
echo "quad_h200 log: ${state_quad}/supervisor.log"
echo "i7_h200 log:   ${state_i7}/supervisor.log"
echo
squeue -u "$USER" -p quad_h200,i7_h200
