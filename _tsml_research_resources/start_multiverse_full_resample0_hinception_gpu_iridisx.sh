#!/bin/bash
# Build the locally available IridisX dataset list for the full multiverse run, split
# it across IridisX's two GPU queues, and start one controller per queue.
#
# a100 (2 concurrent GPUs, no account/QoS) and swarm_a100 (8 concurrent GPUs via the
# ecs account) each get their own controller reading a disjoint half of the dataset
# list. squeue is queried per-partition (see _query_slurm in multiverse_controller.py),
# so a task queued on one partition is invisible to the other controller -- splitting
# the list up front is what keeps the two controllers from racing to submit the same
# task twice. Both controllers still sort their own half smallest-first independently
# (small_datasets_first re-measures on-disk size, it does not just follow file order),
# so alternating lines between the two halves keeps both queues busy with a size mix
# rather than one queue front-loaded with every large problem.

set -euo pipefail

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
repo_dir=$(cd -- "${script_dir}/.." && pwd)
config_a100="${script_dir}/multiverse_full_resample0_hinception_gpu_iridisx_a100.toml"
config_swarm="${script_dir}/multiverse_full_resample0_hinception_gpu_iridisx_swarm.toml"
source_list="${script_dir}/dataset_lists/Multivariate133Classification-MultiverseClean.txt"
supervisor="${script_dir}/run_multiverse_controller.sh"
data_dir="/home/${USER}/Data/Multiverse"
list_dir="/home/${USER}/DataSetLists"
available_list="${list_dir}/MultiverseAvailableGPU-IridisX.txt"
missing_list="${list_dir}/MultiverseUnavailableGPU-IridisX.txt"
excluded_list="${list_dir}/MultiverseExcludedGPU-IridisX.txt"
a100_list="${list_dir}/MultiverseAvailableGPU-IridisX-A100.txt"
swarm_list="${list_dir}/MultiverseAvailableGPU-IridisX-Swarm.txt"
state_dir_a100="/home/${USER}/Results/Multiverse/.controller-full-resample0-hinception-gpu-iridisx-a100"
state_dir_swarm="/home/${USER}/Results/Multiverse/.controller-full-resample0-hinception-gpu-iridisx-swarm"
label_a100="multiverse-full-hinception-iridisx-a100"
label_swarm="multiverse-full-hinception-iridisx-swarm"
python_executable="/home/${USER}/.conda/envs/tsml-eval-gpu/bin/python"
required_branch="ajb/gpu"

# IridisX has neither screen nor tmux. setsid gives the supervisor its own session so
# it is immune to the SIGHUP a login-node logout would otherwise send it; nohup and the
# closed stdin/redirected stdout are belt-and-suspenders for the same purpose. Liveness
# is then confirmed by matching the supervisor's command line with pgrep (the same
# pattern used to replace old copies below), rather than by a terminal multiplexer's
# session list; a PID file is also written for convenience but is not load-bearing.
for command_name in flock git mktemp pkill setsid squeue; do
    if ! command -v "$command_name" >/dev/null 2>&1; then
        echo "ERROR: required command is unavailable: ${command_name}" >&2
        exit 1
    fi
done

for required_file in "$config_a100" "$config_swarm" "$source_list" "$supervisor" \
    "$python_executable"; do
    if [[ ! -f "$required_file" ]]; then
        echo "ERROR: required file not found: ${required_file}" >&2
        exit 1
    fi
done
if [[ ! -x "$python_executable" ]]; then
    echo "ERROR: GPU-environment Python is not executable: ${python_executable}" >&2
    exit 1
fi

actual_branch=$(git -C "$repo_dir" branch --show-current)
if [[ "$actual_branch" != "$required_branch" ]]; then
    echo "ERROR: GPU jobs must run from ${required_branch}; found ${actual_branch}." >&2
    exit 1
fi

if [[ ! -d "$data_dir" ]]; then
    echo "ERROR: Multiverse data directory not found: ${data_dir}" >&2
    exit 1
fi

mkdir -p "$list_dir" "$state_dir_a100" "$state_dir_swarm"
available_tmp=$(mktemp)
missing_tmp=$(mktemp)
excluded_tmp=$(mktemp)
cleanup() {
    rm -f -- "$available_tmp" "$missing_tmp" "$excluded_tmp"
}
trap cleanup EXIT

archive_count=0
available_count=0
excluded_count=0
while IFS= read -r clean_dataset || [[ -n "$clean_dataset" ]]; do
    clean_dataset=${clean_dataset//$'\r'/}
    if [[ -z "$clean_dataset" || "$clean_dataset" == \#* ]]; then
        continue
    fi

    # Results retain the archive's base problem name. The clean-list suffix
    # identifies the exact equal-length/no-missing variant that must be present.
    dataset=${clean_dataset%_eq_nmv}
    dataset=${dataset%_eq}
    dataset=${dataset%_nmv}

    if [[ "$dataset" == DREAM* || "$dataset" == S2Agri-* ]]; then
        printf '%s\n' "$dataset" >> "$excluded_tmp"
        ((excluded_count += 1))
        continue
    fi
    ((archive_count += 1))

    # Aeon's downloaded-dataset check requires the base pair even when a clean
    # variant is selected. Requiring both prevents load_classification from
    # downloading an absent record and guarantees the requested clean variant.
    if [[ -f "${data_dir}/${dataset}/${dataset}_TRAIN.ts" && \
          -f "${data_dir}/${dataset}/${dataset}_TEST.ts" && \
          -f "${data_dir}/${dataset}/${clean_dataset}_TRAIN.ts" && \
          -f "${data_dir}/${dataset}/${clean_dataset}_TEST.ts" ]]; then
        printf '%s\n' "$dataset" >> "$available_tmp"
        ((available_count += 1))
    else
        printf '%s\n' "$dataset" >> "$missing_tmp"
    fi
done < "$source_list"

missing_count=$((archive_count - available_count))
echo "Eligible archive problems: ${archive_count}"
echo "Explicitly excluded:       ${excluded_count}"
echo "Locally available:         ${available_count}"
echo "Unavailable (not queued):  ${missing_count}"
if ((excluded_count)); then
    echo "Excluded datasets:"
    sed 's/^/  /' "$excluded_tmp"
fi
if ((missing_count)); then
    echo "Unavailable datasets:"
    sed 's/^/  /' "$missing_tmp"
fi

if ((available_count == 0)); then
    echo "ERROR: no locally complete eligible datasets were found; refusing to submit." >&2
    exit 1
fi

mv -- "$available_tmp" "$available_list"
mv -- "$missing_tmp" "$missing_list"
mv -- "$excluded_tmp" "$excluded_list"
trap - EXIT

# Alternate lines between the two queues so both get a size mix rather than one
# queue receiving every dataset that happens to sort first.
awk 'NR % 2 == 1' "$available_list" > "$a100_list"
awk 'NR % 2 == 0' "$available_list" > "$swarm_list"
echo "a100 queue:       $(wc -l < "$a100_list") datasets -> ${a100_list}"
echo "swarm_a100 queue: $(wc -l < "$swarm_list") datasets -> ${swarm_list}"

# Replace only earlier copies of these two controllers.
for config in "$config_a100" "$config_swarm"; do
    config_name=$(basename "$config")
    pkill -TERM -f "[r]un_multiverse_controller.sh.*${config_name}" || true
    pkill -TERM -f "[m]ultiverse_controller.py.*${config_name}" || true
done
# Give TERM a moment to land before the new instances try to take the lock files.
sleep 1

cd "$repo_dir"
echo "Checking a100 work without submitting it."
"$python_executable" -u "${script_dir}/multiverse_controller.py" \
    --config "$config_a100" --dry-run --no-email
echo "Checking swarm_a100 work without submitting it."
"$python_executable" -u "${script_dir}/multiverse_controller.py" \
    --config "$config_swarm" --dry-run --no-email

echo "Starting detached a100 controller: ${label_a100}"
setsid nohup flock -n "${state_dir_a100}/supervisor.lock" \
    env PYTHON="$python_executable" \
        MULTIVERSE_CLEAR_PENDING_ON_START=false \
        MULTIVERSE_LOG_DIR="$state_dir_a100" \
    bash "$supervisor" "$config_a100" \
    > "${state_dir_a100}/launcher.out" 2>&1 < /dev/null &
pid_a100=$!
disown "$pid_a100"
echo "$pid_a100" > "${state_dir_a100}/launcher.pid"

echo "Starting detached swarm_a100 controller: ${label_swarm}"
setsid nohup flock -n "${state_dir_swarm}/supervisor.lock" \
    env PYTHON="$python_executable" \
        MULTIVERSE_CLEAR_PENDING_ON_START=false \
        MULTIVERSE_LOG_DIR="$state_dir_swarm" \
    bash "$supervisor" "$config_swarm" \
    > "${state_dir_swarm}/launcher.out" 2>&1 < /dev/null &
pid_swarm=$!
disown "$pid_swarm"
echo "$pid_swarm" > "${state_dir_swarm}/launcher.pid"

sleep 2
# Verify via process-command matching rather than trusting the stored PID alone:
# setsid may or may not fork depending on whether this shell's background job is
# already a process group leader, so the pid in $! is not always the innermost
# process. Matching the actual supervisor/controller command line is what pkill
# above already relies on, so it is the more reliable liveness signal here too.
config_a100_name=$(basename "$config_a100")
config_swarm_name=$(basename "$config_swarm")
if ! pgrep -f "[r]un_multiverse_controller.sh.*${config_a100_name}" >/dev/null; then
    echo "ERROR: a100 controller did not remain running." >&2
    echo "Check ${state_dir_a100}/launcher.out -- another supervisor may already hold its lock file." >&2
    exit 1
fi
if ! pgrep -f "[r]un_multiverse_controller.sh.*${config_swarm_name}" >/dev/null; then
    echo "ERROR: swarm_a100 controller did not remain running." >&2
    echo "Check ${state_dir_swarm}/launcher.out -- another supervisor may already hold its lock file." >&2
    exit 1
fi

echo
echo "Both controllers started."
echo "Available list:   ${available_list}"
echo "Unavailable list: ${missing_list}"
echo "Excluded list:    ${excluded_list}"
echo "a100 controller:       started pid ${pid_a100}, log ${state_dir_a100}/supervisor.log"
echo "swarm_a100 controller: started pid ${pid_swarm}, log ${state_dir_swarm}/supervisor.log"
echo "Stop either with the same pkill pattern this script uses to replace old copies, e.g.:"
echo "  pkill -f 'run_multiverse_controller.sh.*${config_a100_name}'"
echo
echo "Current a100 queue:"
squeue -u "$USER" -p a100
echo
echo "Current swarm_a100 queue:"
squeue -u "$USER" -p swarm_a100
