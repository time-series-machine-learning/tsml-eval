#!/bin/bash

set -euo pipefail

# Run every interval-based regressor over the 63 problem TSER archive.
#
# Scope: 7 regressors x 63 datasets x 30 resamples = 13230 experiments.
#
# The 500 tree variants are used wherever one exists, so the ensemble size
# matches the convention used for the interval classifiers. summary-intervals
# and quant have no 500 variant: summary-intervals already builds a 500 tree
# forest internally, and QUANT has no ensemble size of its own.
#
# Submission model. Each round submits four single node jobs, each running its
# experiments in parallel under staskfarm with one CPU per experiment. Memory is
# requested per CPU, so the memory tier sets how many experiments run in
# parallel on a node. Only four nodes may run at once, so the aim is to keep
# all 192 cores of each of them busy. The batch nodes carry 752 GiB, so at 4 GiB
# a node runs 185 experiments at once and four nodes hold well over 500.
#
# Rounds and recovery. A single 60 hour allocation cannot finish 13230
# experiments, so this script is built to run repeatedly. Each invocation
#
#   1. reconciles what finished, what died, and why,
#   2. escalates the memory tier of anything that ran out of memory,
#   3. resubmits only the resamples that are still missing,
#   4. chains itself as a dependent Slurm job so the next round starts
#      automatically when this round's four nodes finish.
#
# That chaining is what makes the run unattended: nothing needs a human between
# the first pass at 4 GiB and the final high memory retries.
#
# Usage:
#
#   bash run_tser_interval_regressors.sh                 # start the run
#   bash run_tser_interval_regressors.sh --dry-run       # show the plan only
#   bash run_tser_interval_regressors.sh --no-chain      # one round, no chain
#
# --round is set by the chained supervisor job and should not be passed by hand
# except when deliberately resuming a run whose chain was cancelled.
#
# Progress is reported by monitor_tser_interval_regressors.sh.

# ==============================================================================
# Experiment configuration
# ==============================================================================

# Resamples are zero-indexed internally:
# start_fold=1 and max_folds=30 runs resamples 0 to 29.
max_folds=30
start_fold=1

max_num_submitted=500
queue="batch"
max_time="60:00:00"

# The supervisor only reconciles state and resubmits, so it is tiny.
supervisor_time="02:00:00"
supervisor_memory="4G"

username="ajb2u23"
mail="NONE"
mailto="${username}@soton.ac.uk"

# Four concurrent single node jobs, as requested.
node_count=4

# Usable memory and cores on a standard batch node. Both bound how many
# experiments a node runs at once, and at the small tiers cores bind first.
# These match the batch nodes reported by sinfo -p batch -o "%c %m": 192 cores
# and 770000 MB, which is 752 GiB. The budget leaves a little headroom for the
# task farm and the operating system.
node_memory_budget_gib="${node_memory_budget_gib:-740}"
max_cpus_per_node="${max_cpus_per_node:-192}"

# Memory per CPU in GiB. A confirmed out of memory kill, or a run that vanished
# without writing an error, moves that one experiment up a tier for its next
# attempt, so a problem that needs memory climbs to it without holding back the
# rest.
memory_tiers_gib=(4 8 16 32 64 128 256 740)

# Which tier a dataset starts at, chosen from the size of its raw .ts files.
#
# Starting everything at 4 GiB would fill the cores, but the handful of large
# problems would spend a whole round being killed before their first useful
# attempt, and those are the slowest fits in the archive. So the size of the
# data picks the opening tier: the archive is mostly small, and only a few
# problems open high.
#
# On the 2024 archive this puts BIDMC32HR, BIDMC32RR, BIDMC32SpO2 and
# PPGDalia_eq at 16 GiB, eleven mid sized problems at 8 GiB, and the remaining
# forty eight at 4 GiB. 16 GiB is a deliberate bet rather than a ceiling:
# those four hold roughly 350 MiB of float64 per split, so a DrCIF fit over its
# three representations should sit well inside it, and anything that does not
# escalates on its own.
large_dataset_bytes="${large_dataset_bytes:-314572800}"   # 300 MiB
medium_dataset_bytes="${medium_dataset_bytes:-62914560}"  #  60 MiB
large_dataset_start_tier="${large_dataset_start_tier:-3}"
medium_dataset_start_tier="${medium_dataset_start_tier:-2}"

# Safety rails for the unattended chain.
max_rounds="${max_rounds:-40}"
max_attempts_per_experiment="${max_attempts_per_experiment:-10}"
max_failed_attempts="${max_failed_attempts:-3}"

local_path="/iridisfs/home/${username}"

job_name_prefix="tser-interval"
submission_label="TSERIntervals"

generate_train_files="false"
predefined_folds="false"
normalise_data="false"

# The interval-based regressors, 500 trees where a 500 variant exists.
regressors=(
    "tsf-500"
    "rise-500"
    "cif-500"
    "drcif-500"
    "randomintervals-500"
    "summary-intervals"
    "quant"
)

# ==============================================================================
# Repository, data, and result locations
# ==============================================================================

tsml_eval_dir="${local_path}/Code/tsml-eval"
aeon_dir="${local_path}/Code/aeon"
expected_branch="ajb/hc2"

script_file_path="${tsml_eval_dir}/tsml_eval/experiments/regression_experiments.py"

env_name="tsml-eval"
python_path="/home/${username}/.conda/envs/${env_name}/bin/python"

data_dir="${local_path}/Data/TSER"

results_dir="${TSER_INTERVAL_RESULTS_ROOT:-${local_path}/Results/TSERIntervals}"
out_dir="${results_dir}/output"
state_dir="${results_dir}/.tser-interval-state"
numba_cache_dir="${local_path}/Code/.cache/${env_name}"

# The Clean list carries the archive directory names actually on disk, with the
# _nmv and _eq suffixes for the cleaned problems.
dataset_list_file="${tsml_eval_dir}/_tsml_research_resources/dataset_lists/Regression63-MonashExtendedClean.txt"

# ==============================================================================
# Command line
# ==============================================================================

round=1
chain="true"
dry_run="false"

usage() {
    printf '%s\n' \
        "Usage:" \
        "  run_tser_interval_regressors.sh [options]" \
        "" \
        "Options:" \
        "  --round N            Round number; set by the chained job." \
        "  --max-rounds N       Stop chaining after this many rounds." \
        "  --dataset-list FILE  Override the dataset list." \
        "  --no-chain           Submit this round only, do not chain." \
        "  --dry-run            Report the plan without submitting anything." \
        "  -h, --help           Show this help."
}

while (($# > 0)); do
    case "$1" in
        --round)
            if (($# < 2)); then
                echo "ERROR: --round requires a value." >&2
                exit 2
            fi
            round="$2"
            shift 2
            ;;
        --max-rounds)
            if (($# < 2)); then
                echo "ERROR: --max-rounds requires a value." >&2
                exit 2
            fi
            max_rounds="$2"
            shift 2
            ;;
        --dataset-list)
            if (($# < 2)); then
                echo "ERROR: --dataset-list requires a value." >&2
                exit 2
            fi
            dataset_list_file="$2"
            shift 2
            ;;
        --no-chain)
            chain="false"
            shift
            ;;
        --dry-run)
            dry_run="true"
            shift
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo "ERROR: unknown argument: $1" >&2
            usage >&2
            exit 2
            ;;
    esac
done

if ! [[ "${round}" =~ ^[0-9]+$ ]] || ((round < 1)); then
    echo "ERROR: --round must be a positive integer." >&2
    exit 2
fi

if ! [[ "${max_rounds}" =~ ^[0-9]+$ ]] || ((max_rounds < 1)); then
    echo "ERROR: --max-rounds must be a positive integer." >&2
    exit 2
fi

script_path="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/$(basename "${BASH_SOURCE[0]}")"

# ==============================================================================
# Validate configuration
# ==============================================================================

if [[ ! -x "${python_path}" ]]; then
    echo "ERROR: Python executable not found or not executable:"
    echo "  ${python_path}"
    exit 1
fi

if [[ ! -f "${script_file_path}" ]]; then
    echo "ERROR: tsml-eval regression script not found:"
    echo "  ${script_file_path}"
    exit 1
fi

if [[ ! -f "${dataset_list_file}" ]]; then
    echo "ERROR: dataset list not found:"
    echo "  ${dataset_list_file}"
    exit 1
fi

for repository in "${tsml_eval_dir}" "${aeon_dir}"; do
    if [[ ! -d "${repository}/.git" ]]; then
        echo "ERROR: Git checkout not found:"
        echo "  ${repository}"
        exit 1
    fi
done

current_branch=$(git -C "${tsml_eval_dir}" rev-parse --abbrev-ref HEAD)
if [[ "${current_branch}" != "${expected_branch}" ]]; then
    echo "ERROR: tsml-eval is on branch ${current_branch}, expected ${expected_branch}."
    exit 1
fi

if ((start_fold < 1 || max_folds < start_fold)); then
    echo "ERROR: invalid fold range ${start_fold}..${max_folds}."
    exit 1
fi

if ((node_memory_budget_gib < memory_tiers_gib[0])); then
    echo "ERROR: node memory budget is smaller than the first memory tier."
    exit 1
fi

datasets=()
while IFS= read -r line || [[ -n "${line}" ]]; do
    line="${line//$'\r'/}"
    line="${line#"${line%%[![:space:]]*}"}"
    line="${line%"${line##*[![:space:]]}"}"
    if [[ -z "${line}" || "${line:0:1}" == "#" ]]; then
        continue
    fi
    datasets+=("${line}")
done < "${dataset_list_file}"

if ((${#datasets[@]} == 0)); then
    echo "ERROR: no datasets read from ${dataset_list_file}."
    exit 1
fi

declare -A seen_datasets=()
declare -A dataset_bytes=()
missing_data=()
for dataset in "${datasets[@]}"; do
    if [[ -n "${seen_datasets[${dataset}]+present}" ]]; then
        echo "ERROR: duplicate dataset: ${dataset}"
        exit 1
    fi
    seen_datasets["${dataset}"]=1

    train_data="${data_dir}/${dataset}/${dataset}_TRAIN.ts"
    test_data="${data_dir}/${dataset}/${dataset}_TEST.ts"
    if [[ ! -s "${train_data}" || ! -s "${test_data}" ]]; then
        missing_data+=("${dataset}")
        continue
    fi

    train_size=$(stat -c %s "${train_data}")
    test_size=$(stat -c %s "${test_data}")
    dataset_bytes["${dataset}"]=$((train_size + test_size))
done

if ((${#missing_data[@]} > 0)); then
    echo "ERROR: missing or empty raw data under ${data_dir} for:"
    printf '  %s\n' "${missing_data[@]}"
    exit 1
fi

mkdir -p "${results_dir}" "${out_dir}" "${state_dir}" "${numba_cache_dir}"

for regressor in "${regressors[@]}"; do
    mkdir -p "${out_dir}/${regressor}"
done

# ==============================================================================
# Pin the source state for the whole chain
# ==============================================================================

commit_file="${state_dir}/pinned-commits.txt"
tsml_eval_head=$(git -C "${tsml_eval_dir}" rev-parse HEAD)
aeon_head=$(git -C "${aeon_dir}" rev-parse HEAD)

if [[ -f "${commit_file}" ]]; then
    pinned_tsml_eval_commit=$(awk '$1 == "tsml-eval" { print $2 }' "${commit_file}")
    pinned_aeon_commit=$(awk '$1 == "aeon" { print $2 }' "${commit_file}")

    if [[ "${pinned_tsml_eval_commit}" != "${tsml_eval_head}" ||
          "${pinned_aeon_commit}" != "${aeon_head}" ]]; then
        echo "ERROR: a checkout moved after this run started."
        echo "  tsml-eval pinned ${pinned_tsml_eval_commit}, now ${tsml_eval_head}"
        echo "  aeon      pinned ${pinned_aeon_commit}, now ${aeon_head}"
        echo "Results in one run must come from one source state. Either restore"
        echo "the commits, or start a new run with a fresh results root."
        exit 1
    fi
else
    {
        printf 'tsml-eval %s\n' "${tsml_eval_head}"
        printf 'aeon %s\n' "${aeon_head}"
    } > "${commit_file}"
    pinned_tsml_eval_commit="${tsml_eval_head}"
    pinned_aeon_commit="${aeon_head}"
fi

# ==============================================================================
# Convert Boolean options into tsml-eval arguments
# ==============================================================================

generate_train_arg=""
predefined_folds_arg=""
normalise_data_arg=""

if [[ "${generate_train_files,,}" == "true" ]]; then
    generate_train_arg="-tr"
fi

if [[ "${predefined_folds,,}" == "true" ]]; then
    predefined_folds_arg="-pr"
fi

if [[ "${normalise_data,,}" == "true" ]]; then
    normalise_data_arg="-rn"
fi

# ==============================================================================
# Attempt state
# ==============================================================================

# One record per experiment that has been submitted at least once:
#
#   regressor <tab> dataset <tab> resample <tab> tier <tab> attempts
#       <tab> failures <tab> last_reason <tab> last_round
#
# tier indexes memory_tiers_gib from 1. last_reason is the outcome observed for
# the most recent attempt, and is what drives escalation.

attempt_file="${state_dir}/attempts.tsv"
declare -A attempt_tier=()
declare -A attempt_count=()
declare -A failure_count=()
declare -A attempt_reason=()
declare -A attempt_round=()

if [[ -f "${attempt_file}" ]]; then
    while IFS=$'\t' read -r state_regressor state_dataset state_resample \
        state_tier state_attempts state_failures state_reason state_round; do
        if [[ -z "${state_regressor:-}" ]]; then
            continue
        fi
        key="${state_regressor}|${state_dataset}|${state_resample}"
        attempt_tier["${key}"]="${state_tier}"
        attempt_count["${key}"]="${state_attempts}"
        failure_count["${key}"]="${state_failures}"
        attempt_reason["${key}"]="${state_reason}"
        attempt_round["${key}"]="${state_round}"
    done < "${attempt_file}"
fi

# Reconciliation touches every one of the 13230 experiments on every round, so
# these helpers avoid command substitution and any other subshell.
experiment_is_complete() {
    local prefix="${results_dir}/$1/Predictions/$2"

    if [[ ! -s "${prefix}/testResample$3.csv" ]]; then
        return 1
    fi

    if [[ -n "${generate_train_arg}" && ! -s "${prefix}/trainResample$3.csv" ]]; then
        return 1
    fi

    return 0
}

latest_log_result=""
latest_log_for() {
    local regressor="$1"
    local dataset="$2"
    local resample="$3"
    local directory="${out_dir}/${regressor}"
    local latest=""
    local candidate
    local -a candidates

    latest_log_result=""
    if [[ ! -d "${directory}" ]]; then
        return
    fi

    shopt -s nullglob
    candidates=("${directory}"/output-"${dataset}"-"${resample}"-*.txt)
    shopt -u nullglob

    for candidate in "${candidates[@]}"; do
        if [[ -z "${latest}" || "${candidate}" -nt "${latest}" ]]; then
            latest="${candidate}"
        fi
    done

    latest_log_result="${latest}"
}

# Classify the most recent attempt of an experiment that has no result.
#
# OOM      a memory kill is recorded in the log
# FAILED   Python or Slurm reported some other error
# KILLED   the process left a log but no result and no error, which here is
#          nearly always a cgroup memory kill or a wall clock cut, so it is
#          treated as memory suspect and escalated
# NOLOG    nothing ever started, usually the round ended before its turn
#
# The verdict is returned in classify_failure_result rather than on stdout, so
# reconciling a whole round costs no subshells.
classify_failure_result=""
classify_failure() {
    local regressor="$1"
    local dataset="$2"
    local resample="$3"
    local log

    latest_log_for "${regressor}" "${dataset}" "${resample}"
    log="${latest_log_result}"

    if [[ -z "${log}" ]]; then
        classify_failure_result="NOLOG"
        return
    fi

    if grep -Eiq \
        'out[ -]?of[ -]?memory|OUT_OF_MEMORY|oom[_-]kill|Killed process|MemoryError|Cannot allocate memory|std::bad_alloc|Unable to allocate' \
        "${log}"; then
        classify_failure_result="OOM"
        return
    fi

    if grep -Eiq \
        'Traceback \(most recent call last\)|Segmentation fault|^ERROR:|slurmstepd: error:|Exception:' \
        "${log}"; then
        classify_failure_result="FAILED"
        return
    fi

    classify_failure_result="KILLED"
}

# ==============================================================================
# Reconcile the previous round and choose this round's memory tiers
# ==============================================================================

max_tier=${#memory_tiers_gib[@]}

# The opening tier for a dataset that has never been attempted.
declare -A dataset_start_tier=()
for dataset in "${datasets[@]}"; do
    if ((dataset_bytes[${dataset}] > large_dataset_bytes)); then
        dataset_start_tier["${dataset}"]="${large_dataset_start_tier}"
    elif ((dataset_bytes[${dataset}] > medium_dataset_bytes)); then
        dataset_start_tier["${dataset}"]="${medium_dataset_start_tier}"
    else
        dataset_start_tier["${dataset}"]=1
    fi
done

pending_keys=()
declare -A pending_tier=()
dead_keys=()
completed_total=0
oom_escalated=0

for regressor in "${regressors[@]}"; do
    for dataset in "${datasets[@]}"; do
        for ((resample = start_fold - 1; resample < max_folds; resample++)); do
            key="${regressor}|${dataset}|${resample}"

            if experiment_is_complete "${regressor}" "${dataset}" "${resample}"; then
                completed_total=$((completed_total + 1))
                if [[ -n "${attempt_reason[${key}]+present}" ]]; then
                    attempt_reason["${key}"]="COMPLETE"
                fi
                continue
            fi

            tier="${attempt_tier[${key}]-${dataset_start_tier[${dataset}]}}"
            attempts="${attempt_count[${key}]-0}"
            failures="${failure_count[${key}]-0}"
            reason="${attempt_reason[${key}]-}"

            if [[ "${reason}" == "DEAD" ]]; then
                dead_keys+=("${key}")
                continue
            fi

            # Only reconcile experiments submitted in an earlier round.
            # Anything submitted in this round is still in flight.
            if ((attempts > 0)) &&
               [[ "${attempt_round[${key}]-0}" != "${round}" ]]; then
                classify_failure "${regressor}" "${dataset}" "${resample}"
                reason="${classify_failure_result}"

                case "${reason}" in
                    OOM|KILLED)
                        if ((tier < max_tier)); then
                            tier=$((tier + 1))
                            oom_escalated=$((oom_escalated + 1))
                        fi
                        ;;
                    FAILED)
                        failures=$((failures + 1))
                        ;;
                esac

                attempt_tier["${key}"]="${tier}"
                failure_count["${key}"]="${failures}"
                attempt_reason["${key}"]="${reason}"
            fi

            # Retire experiments another attempt cannot rescue.
            if ((attempts >= max_attempts_per_experiment)) ||
               ((failures >= max_failed_attempts)); then
                attempt_reason["${key}"]="DEAD"
                dead_keys+=("${key}")
                continue
            fi

            pending_keys+=("${key}")
            pending_tier["${key}"]="${tier}"
        done
    done
done

total_experiments=$((${#regressors[@]} * ${#datasets[@]} * (max_folds - start_fold + 1)))

echo "TSER interval regressor run - round ${round} of at most ${max_rounds}"
echo "Results:           ${results_dir}"
echo "Data:              ${data_dir}"
echo "Regressors:        ${#regressors[@]}"
echo "Datasets:          ${#datasets[@]}"
echo "Resamples:         $((max_folds - start_fold + 1))"
echo "Experiments:       ${total_experiments}"
echo "Complete:          ${completed_total}"
echo "Pending:           ${#pending_keys[@]}"
echo "Escalated:         ${oom_escalated} (memory kill or silent death)"
echo "Retired:           ${#dead_keys[@]}"
echo "tsml-eval commit:  ${pinned_tsml_eval_commit}"
echo "aeon commit:       ${pinned_aeon_commit}"
echo

if ((${#pending_keys[@]} == 0)); then
    echo "Nothing left to run."
    if ((${#dead_keys[@]} > 0)); then
        echo "Retired experiments that never produced a result:"
        printf '  %s\n' "${dead_keys[@]}"
    fi
    exit 0
fi

# ==============================================================================
# Allocate the four nodes across the memory tiers present in this round
# ==============================================================================

declare -A tier_pending_count=()
for key in "${pending_keys[@]}"; do
    tier="${pending_tier[${key}]}"
    tier_pending_count["${tier}"]=$(( ${tier_pending_count[${tier}]-0} + 1 ))
done

active_tiers=()
for ((tier = 1; tier <= max_tier; tier++)); do
    if (( ${tier_pending_count[${tier}]-0} > 0 )); then
        active_tiers+=("${tier}")
    fi
done

# How many experiments a node runs at once at each tier. A 4 GiB node holds the
# full 185 experiments a node can hold, a 16 GiB node holds 46, so the same
# queue length means very different waits at different tiers.
declare -A tier_cpus=()
for tier in "${active_tiers[@]}"; do
    tier_cpus["${tier}"]=$((node_memory_budget_gib / memory_tiers_gib[tier - 1]))
    if ((tier_cpus[${tier}] > max_cpus_per_node)); then
        tier_cpus["${tier}"]="${max_cpus_per_node}"
    fi
    if ((tier_cpus[${tier}] < 1)); then
        tier_cpus["${tier}"]=1
    fi
done

# Every tier with work gets at least one node. Spare nodes then go to whichever
# tier faces the longest sequential queue per task slot, so the four nodes are
# balanced by the time they will take rather than by experiment count.
declare -A tier_slots=()
remaining_slots=${node_count}
for tier in "${active_tiers[@]}"; do
    tier_slots["${tier}"]=1
    remaining_slots=$((remaining_slots - 1))
done

while ((remaining_slots > 0)); do
    best_tier=""
    best_load=-1
    for tier in "${active_tiers[@]}"; do
        load=$((
            ${tier_pending_count[${tier}]} /
            (${tier_slots[${tier}]} * ${tier_cpus[${tier}]})
        ))
        if ((load > best_load)); then
            best_load=${load}
            best_tier="${tier}"
        fi
    done
    if [[ -z "${best_tier}" ]]; then
        break
    fi
    tier_slots["${best_tier}"]=$(( ${tier_slots[${best_tier}]} + 1 ))
    remaining_slots=$((remaining_slots - 1))
done

# ==============================================================================
# Submission
# ==============================================================================

run_id=$(date +%Y%m%d%H%M%S)
submission_dir="${results_dir}/batch-submissions/${run_id}-round${round}"
mkdir -p "${submission_dir}"

total_commands=0
submitted_job_ids=()

wait_for_queue_slot() {
    local num_jobs

    while true; do
        num_jobs=$(
            squeue \
                --noheader \
                --user="${username}" \
                --partition="${queue}" \
                --states=RUNNING,PENDING |
                wc -l
        )

        if ((num_jobs < max_num_submitted)); then
            break
        fi

        echo "Waiting 60 seconds: ${num_jobs} jobs are running or pending."
        sleep 60
    done
}

write_command() {
    local regressor="$1"
    local dataset="$2"
    local resample="$3"
    local batch_id="$4"
    local command_file="$5"
    local experiment_output
    local command_line
    local -a command

    experiment_output="${out_dir}/${regressor}/output-${dataset}-${resample}-${batch_id}.txt"

    command=(
        "${python_path}"
        -u
        "${script_file_path}"
        "${data_dir}"
        "${results_dir}"
        "${regressor}"
        "${dataset}"
        "${resample}"
    )

    if [[ -n "${generate_train_arg}" ]]; then
        command+=("${generate_train_arg}")
    fi
    if [[ -n "${predefined_folds_arg}" ]]; then
        command+=("${predefined_folds_arg}")
    fi
    if [[ -n "${normalise_data_arg}" ]]; then
        command+=("${normalise_data_arg}")
    fi

    printf -v command_line '%q ' "${command[@]}"
    printf '%s> %q 2>&1\n' \
        "${command_line}" \
        "${experiment_output}" \
        >> "${command_file}"
}

submit_node_job() {
    local batch_label="$1"
    local memory_gib="$2"
    local command_file="$3"
    local cmd_count="$4"
    local batch_id="$5"
    local submission_file="${submission_dir}/generatedSubmissionFile-${batch_id}.sub"
    local cpu_count
    local max_cpus_to_use
    local sbatch_output
    local job_id

    max_cpus_to_use=$((node_memory_budget_gib / memory_gib))
    if ((max_cpus_to_use > max_cpus_per_node)); then
        max_cpus_to_use=${max_cpus_per_node}
    fi
    if ((max_cpus_to_use < 1)); then
        max_cpus_to_use=1
    fi
    cpu_count=$((cmd_count < max_cpus_to_use ? cmd_count : max_cpus_to_use))

    cat > "${submission_file}" <<SUB
#!/bin/bash
#SBATCH --mail-type=${mail}
#SBATCH --mail-user=${mailto}
#SBATCH --job-name=${job_name_prefix}-r${round}-${batch_label}
#SBATCH --partition=${queue}
#SBATCH --time=${max_time}
#SBATCH --output=${submission_dir}/%A-${batch_id}.out
#SBATCH --error=${submission_dir}/%A-${batch_id}.err
#SBATCH --nodes=1
#SBATCH --ntasks=${cpu_count}
#SBATCH --mem-per-cpu=${memory_gib}G

. /etc/profile
set -e

cd "${tsml_eval_dir}" || exit 1

unset PYTHONHOME
export PYTHONNOUSERSITE=1
export PYTHONPATH="${aeon_dir}:${tsml_eval_dir}"

export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export NUMBA_NUM_THREADS=1
export LOKY_MAX_CPU_COUNT=1
export PYTHONUNBUFFERED=1

export NUMBA_CACHE_DIR="${numba_cache_dir}"
mkdir -p "\${NUMBA_CACHE_DIR}"

current_tsml_eval_commit=\$(git -C "${tsml_eval_dir}" rev-parse HEAD)
current_aeon_commit=\$(git -C "${aeon_dir}" rev-parse HEAD)

if [[ "\${current_tsml_eval_commit}" != "${pinned_tsml_eval_commit}" ]]; then
    echo "ERROR: tsml-eval changed after submission."
    echo "Expected: ${pinned_tsml_eval_commit}"
    echo "Current:  \${current_tsml_eval_commit}"
    exit 1
fi

if [[ "\${current_aeon_commit}" != "${pinned_aeon_commit}" ]]; then
    echo "ERROR: aeon changed after submission."
    echo "Expected: ${pinned_aeon_commit}"
    echo "Current:  \${current_aeon_commit}"
    exit 1
fi

echo "Round:             ${round}"
echo "Batch:             ${batch_label}"
echo "Memory per task:   ${memory_gib} GiB"
echo "Host:              \$(hostname)"
echo "Slurm job ID:      \${SLURM_JOB_ID}"
echo "Allocated tasks:   \${SLURM_NTASKS}"
echo "Command count:     ${cmd_count}"
echo "tsml-eval commit:  \${current_tsml_eval_commit}"
echo "aeon commit:       \${current_aeon_commit}"
echo "Command file:      ${command_file}"
echo

staskfarm "${command_file}"
SUB

    if [[ "${dry_run}" == "true" ]]; then
        echo "${batch_label}: would submit ${cmd_count} command(s) on ${cpu_count} CPU(s) at ${memory_gib} GiB each."
        return
    fi

    wait_for_queue_slot

    sbatch_output=$(sbatch "${submission_file}")
    job_id="${sbatch_output##* }"
    submitted_job_ids+=("${job_id}")
    echo "${batch_label}: ${cmd_count} command(s) on ${cpu_count} CPU(s) at ${memory_gib} GiB each -> ${sbatch_output}"
}

# Build the per node command files. Within a tier, experiments are sorted by raw
# data size and dealt to the least loaded node, so the four nodes finish close
# together instead of one node holding every large problem.
for tier in "${active_tiers[@]}"; do
    memory_gib="${memory_tiers_gib[$((tier - 1))]}"
    slots="${tier_slots[${tier}]}"

    slot_files=()
    slot_counts=()
    slot_loads=()
    slot_ids=()

    for ((slot = 0; slot < slots; slot++)); do
        batch_id="${run_id}-${submission_label}-r${round}-mem${memory_gib}-node$((slot + 1))"
        slot_ids[slot]="${batch_id}"
        slot_files[slot]="${submission_dir}/generatedCommandList-${batch_id}.txt"
        : > "${slot_files[slot]}"
        slot_counts[slot]=0
        slot_loads[slot]=0
    done

    while IFS=$'\t' read -r weight key; do
        IFS='|' read -r command_regressor command_dataset command_resample <<< "${key}"

        best_slot=0
        best_load=-1
        for ((slot = 0; slot < slots; slot++)); do
            if ((best_load < 0 || slot_loads[slot] < best_load)); then
                best_load=${slot_loads[slot]}
                best_slot=${slot}
            fi
        done

        write_command \
            "${command_regressor}" "${command_dataset}" "${command_resample}" \
            "${slot_ids[best_slot]}" "${slot_files[best_slot]}"

        slot_loads[best_slot]=$((slot_loads[best_slot] + weight))
        slot_counts[best_slot]=$((slot_counts[best_slot] + 1))

        attempt_tier["${key}"]="${tier}"
        attempt_count["${key}"]=$(( ${attempt_count[${key}]-0} + 1 ))
        attempt_reason["${key}"]="SUBMITTED"
        attempt_round["${key}"]="${round}"
        failure_count["${key}"]="${failure_count[${key}]-0}"
    done < <(
        for key in "${pending_keys[@]}"; do
            if [[ "${pending_tier[${key}]}" != "${tier}" ]]; then
                continue
            fi
            IFS='|' read -r sort_regressor sort_dataset sort_resample <<< "${key}"
            printf '%s\t%s\n' "${dataset_bytes[${sort_dataset}]}" "${key}"
        done | sort -k1,1nr
    )

    for ((slot = 0; slot < slots; slot++)); do
        if ((slot_counts[slot] == 0)); then
            rm -f "${slot_files[slot]}"
            continue
        fi
        submit_node_job \
            "mem${memory_gib}-node$((slot + 1))" \
            "${memory_gib}" \
            "${slot_files[slot]}" \
            "${slot_counts[slot]}" \
            "${slot_ids[slot]}"
        total_commands=$((total_commands + slot_counts[slot]))
    done
done

# ==============================================================================
# Persist attempt state
# ==============================================================================

if [[ "${dry_run}" != "true" ]]; then
    tmp_attempt_file="${attempt_file}.tmp"
    : > "${tmp_attempt_file}"
    for key in "${!attempt_count[@]}"; do
        IFS='|' read -r state_regressor state_dataset state_resample <<< "${key}"
        printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
            "${state_regressor}" "${state_dataset}" "${state_resample}" \
            "${attempt_tier[${key}]-1}" \
            "${attempt_count[${key}]-0}" \
            "${failure_count[${key}]-0}" \
            "${attempt_reason[${key}]-UNKNOWN}" \
            "${attempt_round[${key}]-0}" \
            >> "${tmp_attempt_file}"
    done
    mv "${tmp_attempt_file}" "${attempt_file}"
fi

# ==============================================================================
# Chain the next round
# ==============================================================================

if [[ "${dry_run}" == "true" ]]; then
    echo
    echo "Dry run: nothing submitted. Submission files are in ${submission_dir}."
    exit 0
fi

echo
echo "Round ${round}: submitted ${#submitted_job_ids[@]} node job(s), ${total_commands} command(s)."

if [[ "${chain}" == "true" ]] && ((round < max_rounds)) &&
   ((${#submitted_job_ids[@]} > 0)); then
    next_round=$((round + 1))
    dependency=$(IFS=:; printf '%s' "${submitted_job_ids[*]}")
    supervisor_file="${submission_dir}/generatedSupervisor-round${next_round}.sub"

    # afterany, not afterok: a node that dies is exactly the case the next round
    # has to reconcile and retry at a higher memory tier.
    cat > "${supervisor_file}" <<SUP
#!/bin/bash
#SBATCH --mail-type=${mail}
#SBATCH --mail-user=${mailto}
#SBATCH --job-name=${job_name_prefix}-supervisor-r${next_round}
#SBATCH --partition=${queue}
#SBATCH --time=${supervisor_time}
#SBATCH --output=${submission_dir}/%A-supervisor-round${next_round}.out
#SBATCH --error=${submission_dir}/%A-supervisor-round${next_round}.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=${supervisor_memory}
#SBATCH --dependency=afterany:${dependency}

. /etc/profile
set -e

bash "${script_path}" \\
    --round ${next_round} \\
    --max-rounds ${max_rounds} \\
    --dataset-list "${dataset_list_file}"
SUP

    supervisor_output=$(sbatch "${supervisor_file}")
    echo "Next round chained: ${supervisor_output}"
    echo "It starts once jobs ${dependency//:/, } have all finished."
else
    if [[ "${chain}" != "true" ]]; then
        echo "Chaining disabled: run the script again to continue."
    elif ((round >= max_rounds)); then
        echo "Round limit ${max_rounds} reached: no further rounds are chained."
    fi
fi

echo
echo "Results:     ${results_dir}"
echo "Submissions: ${submission_dir}"
echo "Monitor:     bash monitor_tser_interval_regressors.sh --details"
