#!/bin/bash

set -euo pipefail

# Fill every remaining gap in the UCR classification results, so that each
# estimator listed below has 30 resamples on all 112 problems of the clean UCR
# archive.
#
# Scope was taken from a full inventory of the results tree (213,430 prediction
# files, none zero length). Thirteen estimators are short of 30 x 112; every
# other estimator in the tree is already complete and is not listed here, so it
# is never touched. The gaps total 9,200 test files:
#
#   BaggedQuant           1    Wafer resample 27 (its train file is missing too)
#   SharedDrCIF-min       2    Beef resamples 22 and 23
#   RIST                 30    StarLightCurves, never run
#   pulsar-et            30    Beef
#   pulsar-nosel         90    ArrowHead, Beef, Computers
#   pulsar-ridge         90    Beef, CBF, Computers
#   pulsar-noar         120    Beef, BirdChicken, Coffee, CricketY
#   pulsar-nopool       181    ArrowHead, Beef, CBF, CinCECGTorso, Computers,
#                              CricketY
#   SharedDrCIF-nopgram 2806   only 19 of 112 problems ever ran
#   CBOSS              1440    truncated from MoteStrain onwards
#   WEASEL_V2          1440    same truncation
#   MrSEQL             1470    same truncation
#   BOSS               1500    same truncation
#
# The four dictionary baselines all stop at the same point in the alphabet,
# which is one interrupted run rather than a set of individual failures. The
# pulsar ablations are missing overlapping problems (Beef appears in five of
# them), so those are more likely a systematic failure on those problems than
# random attrition; if they die again at the top memory tier the run retires
# them rather than looping.
#
# Nothing here is recomputed. Reconciliation reads the results tree on every
# round and submits only the (estimator, dataset, resample) triples whose files
# are absent, so declaring the full 13 x 112 x 30 grid is safe: the 34,480
# experiments that already exist are skipped.
#
# ---------------------------------------------------------------------------
#
# Estimators, results locations, and train files. Each estimator keeps the
# results directory and the train-file convention it already has on disk, so
# this run extends the existing directories rather than creating parallel ones.
# The four dictionary baselines and BaggedQuant carry train files; the rest are
# test-only, matching what is already there.
#
# All thirteen estimator keys resolve on the ajb/drcif branch, which is the only
# branch carrying the _wip classes (BaggedQUANT, SharedDrCIF and the PULSAR
# ablations) together with the aeon estimators. ajb/hc2 does NOT register the
# _wip variants, so this run cannot be driven from it.
#
# Submission model. Each round submits up to four single node jobs, each running
# its experiments in parallel under staskfarm with one CPU per experiment. Memory
# is requested per CPU, so the memory tier sets how many experiments run in
# parallel on a node. Only four nodes may run at once, so the aim is to keep all
# 192 cores of each of them busy. The batch nodes carry 752 GiB but Slurm caps
# one job at 634 GiB of it, so at 4 GiB a node runs 157 experiments at once and
# four nodes hold over 600.
#
# Rounds and recovery. A single 60 hour allocation cannot finish 9,200 heavy
# experiments, so this script is built to run repeatedly. Each invocation
#
#   1. reconciles what finished, what died, and why,
#   2. escalates the memory tier of anything that ran out of memory,
#   3. resubmits only the resamples that are still missing,
#   4. arms one dependent successor per node job it submits, so each node
#      refills its own freed slot the moment it finishes rather than waiting
#      for a whole round of four to end.
#
# That continuous refill keeps the nodes saturated and the run unattended.
#
# Usage:
#
#   bash run_ucr_completion.sh --dry-run       # ALWAYS do this first
#   bash run_ucr_completion.sh                 # start the run
#   bash run_ucr_completion.sh --no-chain      # one round, no chain
#   bash run_ucr_completion.sh --round 2 --no-chain
#                                              # fill spare nodes early
#
# Run --dry-run before anything else and check that it reports about 9,200
# pending. That number is the gap measured on the results archive; if the dry
# run reports far more, the results tree on this cluster is not the one that was
# inventoried and the run would recompute work that already exists elsewhere.
#
# --round is set by the chained supervisor job and should not be passed by hand
# except when deliberately resuming a run whose chain was cancelled.

# ==============================================================================
# Experiment configuration
# ==============================================================================

# Resamples are zero-indexed internally:
# start_fold=1 and max_folds=30 runs resamples 0 to 29.
max_folds=30
start_fold=1

queue="batch"
max_time="60:00:00"

# The supervisor only reconciles state and resubmits, so it is tiny.
supervisor_time="02:00:00"
supervisor_memory="4G"

username="ajb2u23"
mailto="${mailto:-${username}@soton.ac.uk}"

# Slurm mail on the node jobs would be four messages a round with nothing in
# them but a job state, so it stays off.
mail="NONE"
supervisor_mail="FAIL"

# Off by default. A per round mail would be noise; the summary is still written
# to the state directory every round either way.
email_updates="${email_updates:-false}"

# Four concurrent single node jobs.
node_count=4

# Usable memory and cores on a standard batch node. The batch nodes report 192
# cores and 770000 MB, but the partition sets MaxMemPerNode=650000, which is
# 634 GiB, and a job asking for more than that is rejected rather than queued.
# The budget stays just under it.
#
# The nodes are shared, so a request near the cap only starts on a node that is
# nearly empty. If rounds sit pending for long, lower this for the invocation to
# start sooner on a partly used node:
#
#   node_memory_budget_gib=450 bash run_ucr_pulsar_classifier.sh
node_memory_budget_gib="${node_memory_budget_gib:-630}"
max_cpus_per_node="${max_cpus_per_node:-192}"

# Memory per CPU in GiB. A confirmed out of memory kill, or a run that vanished
# without writing an error, moves that one experiment up a tier for its next
# attempt.
memory_tiers_gib=(4 8 16 32 64 128 256 620)

# Which tier a dataset starts at, chosen from the size of its raw .ts files.
# The UCR archive is mostly small, so most problems open at 4 GiB and only the
# few large ones open higher; anything under-provisioned escalates on its own.
large_dataset_bytes="${large_dataset_bytes:-314572800}"   # 300 MiB
medium_dataset_bytes="${medium_dataset_bytes:-62914560}"  #  60 MiB
large_dataset_start_tier="${large_dataset_start_tier:-3}"
medium_dataset_start_tier="${medium_dataset_start_tier:-2}"

# Safety rails for the unattended chain.
max_rounds="${max_rounds:-500}"
max_attempts_per_experiment="${max_attempts_per_experiment:-10}"
max_failed_attempts="${max_failed_attempts:-3}"

local_path="/iridisfs/home/${username}"

job_name_prefix="ucr-completion"
submission_label="UCRCompletion"
workflow_label="results completion"

predefined_folds="false"
normalise_data="false"

# One line per estimator:
#
#   <classifier key> | <results subdirectory> | <train files: yes|no>
#
# The classifier key is passed to tsml-eval unchanged, and tsml-eval names the
# results directory after it, so the key must be spelled exactly as the existing
# directory on disk. Lookup itself is case-insensitive.
#
# The results subdirectory is relative to results_root and is where that
# estimator's results already live. Keeping each estimator in its own existing
# directory is what allows this one run to fill gaps that are scattered across
# IntervalBased, DictionaryBased, Hybrid, TestOnly and TestOnlyExperimental.
#
# Train files follow what each estimator already has. Turning them on for an
# estimator that has none would leave that estimator permanently incomplete by
# this script's own definition, because its existing test-only resamples would
# never satisfy the train check and would all be recomputed.
experiment_spec=(
    "BOSS|DictionaryBased|yes"
    "CBOSS|DictionaryBased|yes"
    "MrSEQL|DictionaryBased|yes"
    "WEASEL_V2|DictionaryBased|yes"
    "BaggedQuant|IntervalBased|yes"
    "RIST|Hybrid|no"
    "SharedDrCIF-min|IntervalBased/TestOnlyExperimental|no"
    "SharedDrCIF-nopgram|IntervalBased/TestOnlyExperimental|no"
    "pulsar-et|IntervalBased/TestOnlyExperimental|no"
    "pulsar-noar|IntervalBased/TestOnlyExperimental|no"
    "pulsar-nopool|IntervalBased/TestOnlyExperimental|no"
    "pulsar-nosel|IntervalBased/TestOnlyExperimental|no"
    "pulsar-ridge|IntervalBased/TestOnlyExperimental|no"
)

classifiers=()
declare -A classifier_results_dir=()
declare -A classifier_train_arg=()
for spec_line in "${experiment_spec[@]}"; do
    IFS='|' read -r spec_name spec_subdir spec_train <<< "${spec_line}"
    if [[ -z "${spec_name}" || -z "${spec_subdir}" || -z "${spec_train}" ]]; then
        echo "ERROR: malformed experiment_spec entry: ${spec_line}" >&2
        exit 2
    fi
    if [[ -n "${classifier_results_dir[${spec_name}]+present}" ]]; then
        echo "ERROR: duplicate classifier in experiment_spec: ${spec_name}" >&2
        exit 2
    fi
    classifiers+=("${spec_name}")
    classifier_results_dir["${spec_name}"]="${spec_subdir}"
    case "${spec_train,,}" in
        yes) classifier_train_arg["${spec_name}"]="-tr" ;;
        no)  classifier_train_arg["${spec_name}"]="" ;;
        *)
            echo "ERROR: train field must be yes or no: ${spec_line}" >&2
            exit 2
            ;;
    esac
done

# ==============================================================================
# Repository, data, and result locations
# ==============================================================================

tsml_eval_dir="${local_path}/Code/tsml-eval"
aeon_dir="${local_path}/Code/aeon"

# The branch that registers all thirteen estimators. This is a warning rather
# than a hard failure: the batch scripts themselves live in tsml-eval and get
# corrected while a run is in flight, and a refusal to continue there strands the
# queue. The estimator keys are resolved for real below, which is the check that
# actually matters.
expected_branch="${expected_branch:-ajb/drcif}"

script_file_path="${tsml_eval_dir}/tsml_eval/experiments/classification_experiments.py"

env_name="tsml-eval"
python_path="/home/${username}/.conda/envs/${env_name}/bin/python"

# UCR datasets live one directory per problem under here:
#   ${data_dir}/${dataset}/${dataset}_TRAIN.ts
# Override with data_dir=... if the archive sits elsewhere on the cluster.
data_dir="${data_dir:-${local_path}/Data/UCR}"

# Every estimator writes under its own subdirectory of this root, named by its
# experiment_spec entry.
results_root="${UCR_RESULTS_ROOT:-${local_path}/Results/UCR}"
out_dir="${results_root}/output-completion"
state_dir="${results_root}/.ucr-completion-state"
numba_cache_dir="${local_path}/Code/.cache/${env_name}"
shared_runner_lock="${UCR_COMPLETION_RUNNER_LOCK:-${results_root}/.ucr-completion-runner.lock}"

# Absolute results directory for each estimator, resolved once. tsml-eval
# appends the classifier name itself, so this is the parent.
declare -A classifier_results_abs=()
for spec_name in "${classifiers[@]}"; do
    classifier_results_abs["${spec_name}"]="${results_root}/${classifier_results_dir[${spec_name}]}"
done

dataset_list_file="${tsml_eval_dir}/_tsml_research_resources/dataset_lists/UnivariateClassification112-UCR2018Clean.txt"

# ==============================================================================
# Command line
# ==============================================================================

round=1
chain="true"
dry_run="false"

usage() {
    printf '%s\n' \
        "Usage:" \
        "  run_ucr_completion.sh [options]" \
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
    echo "ERROR: tsml-eval classification script not found:"
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
    echo "WARNING: tsml-eval is on branch ${current_branch}, expected ${expected_branch}."
    echo "         Every estimator key is resolved below; this is informational."
fi

# A misspelled or unregistered estimator key would otherwise fail 3360 times
# inside the task farm, so construct all of them once, here.
unresolved_classifiers=()
for classifier in "${classifiers[@]}"; do
    if ! PYTHONPATH="${aeon_dir}:${tsml_eval_dir}" PYTHONNOUSERSITE=1 \
        "${python_path}" -c '
import sys
from tsml_eval.experiments._get_classifier import get_classifier_by_name

get_classifier_by_name(sys.argv[1], random_state=0)
' "${classifier}" >/dev/null 2>&1; then
        unresolved_classifiers+=("${classifier}")
    fi
done
if ((${#unresolved_classifiers[@]} > 0)); then
    echo "ERROR: these classifier keys do not resolve in this checkout:" >&2
    printf '  %s\n' "${unresolved_classifiers[@]}" >&2
    echo "Branch ${current_branch} is missing them; ${expected_branch} carries all thirteen." >&2
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

# Slurm rejects rather than queues a job asking for more than MaxMemPerNode, so
# any tier above the budget is clamped to it.
tier_ceiling_note=""
for ((tier_index = 0; tier_index < ${#memory_tiers_gib[@]}; tier_index++)); do
    if ((memory_tiers_gib[tier_index] > node_memory_budget_gib)); then
        memory_tiers_gib[tier_index]="${node_memory_budget_gib}"
        tier_ceiling_note="one or more tiers clamped to ${node_memory_budget_gib} GiB"
    fi
done

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
    echo "Set data_dir=... if the UCR archive lives elsewhere on the cluster."
    exit 1
fi

mkdir -p \
    "${results_root}" \
    "${out_dir}" \
    "${state_dir}" \
    "$(dirname "${shared_runner_lock}")" \
    "${numba_cache_dir}"

# Serialize reconciliation and submission so a chained supervisor and a manual
# spare-node refill that become runnable together do not race and duplicate work.
for required_command in flock sacct sbatch scontrol squeue sha256sum; do
    if ! command -v "${required_command}" >/dev/null 2>&1; then
        echo "ERROR: ${required_command} is required for submission rounds." >&2
        exit 1
    fi
done
exec 9> "${shared_runner_lock}"
if ! flock -w 300 9; then
    echo "ERROR: another submission round held the shared lock for five minutes." >&2
    exit 1
fi

for classifier in "${classifiers[@]}"; do
    mkdir -p "${classifier_results_abs[${classifier}]}/${classifier}"
    mkdir -p "${out_dir}/${classifier}"
done

# ==============================================================================
# Provenance
# ==============================================================================

# aeon is pinned hard: resamples of one estimator must not be split across two
# aeon versions, and aeon is not something this run edits, so a change there
# means something moved underneath it.
#
# The estimator sources are recorded but NOT enforced. These batch scripts and
# the _wip estimators live in the same repository, and refusing to continue after
# an operational fix strands the queue for as long as nobody notices. A change is
# reported loudly on every round instead.
tsml_eval_head=$(git -C "${tsml_eval_dir}" rev-parse HEAD)
aeon_head=$(git -C "${aeon_dir}" rev-parse HEAD)

estimator_source_files=(
    "${tsml_eval_dir}/tsml_eval/experiments/_get_classifier.py"
)
shopt -s nullglob
estimator_source_files+=("${tsml_eval_dir}"/tsml_eval/_wip/classification/*.py)
shopt -u nullglob
for source_file in "${estimator_source_files[@]}"; do
    if [[ ! -s "${source_file}" ]]; then
        echo "ERROR: estimator source file is missing or empty: ${source_file}" >&2
        exit 1
    fi
done
estimator_source_hash=$(
    sha256sum "${estimator_source_files[@]}" | sha256sum | cut -d ' ' -f 1
)
pinned_source_file="${state_dir}/pinned-sources.txt"
estimator_source_note=""

if [[ -f "${pinned_source_file}" ]]; then
    pinned_aeon_commit=$(awk '$1 == "aeon" { print $2 }' "${pinned_source_file}")
    pinned_estimator_hash=$(awk '$1 == "estimators" { print $2 }' "${pinned_source_file}")
    if [[ "${pinned_aeon_commit}" != "${aeon_head}" ]]; then
        echo "ERROR: aeon changed after this run started."
        echo "  aeon pinned ${pinned_aeon_commit}, now ${aeon_head}"
        echo "Check out the pinned commit, or delete ${pinned_source_file} to"
        echo "accept the new one for the resamples that are still outstanding."
        exit 1
    fi
    if [[ "${pinned_estimator_hash}" != "${estimator_source_hash}" ]]; then
        estimator_source_note="estimator sources changed since round 1 (pinned ${pinned_estimator_hash:0:12}, now ${estimator_source_hash:0:12})"
    fi
else
    printf 'aeon %s\nestimators %s\n' \
        "${aeon_head}" "${estimator_source_hash}" > "${pinned_source_file}"
    pinned_aeon_commit="${aeon_head}"
    pinned_estimator_hash="${estimator_source_hash}"
fi

# Record the whole checkout revisions as additional provenance.
printf 'tsml-eval %s  aeon %s  round %s\n' \
    "${tsml_eval_head}" "${aeon_head}" "${round}" \
    >> "${state_dir}/commits.txt"

# ==============================================================================
# Convert Boolean options into tsml-eval arguments
# ==============================================================================

predefined_folds_arg=""
normalise_data_arg=""

if [[ "${predefined_folds,,}" == "true" ]]; then
    predefined_folds_arg="-pr"
fi
if [[ "${normalise_data,,}" == "true" ]]; then
    normalise_data_arg="-rn"
fi

# ==============================================================================
# Attempt state
# ==============================================================================

# One record per experiment submitted at least once:
#
#   classifier <tab> dataset <tab> resample <tab> tier <tab> attempts
#       <tab> failures <tab> last_reason <tab> last_round <tab> last_job_id
#
# tier indexes memory_tiers_gib from 1. last_reason is the outcome observed for
# the most recent attempt, and is what drives escalation.

attempt_file="${state_dir}/attempts.tsv"
declare -A attempt_tier=()
declare -A attempt_count=()
declare -A failure_count=()
declare -A attempt_reason=()
declare -A attempt_round=()
declare -A attempt_job_id=()

if [[ -f "${attempt_file}" ]]; then
    while IFS=$'\t' read -r state_classifier state_dataset state_resample \
        state_tier state_attempts state_failures state_reason state_round \
        state_job_id; do
        if [[ -z "${state_classifier:-}" ]]; then
            continue
        fi
        key="${state_classifier}|${state_dataset}|${state_resample}"
        attempt_tier["${key}"]="${state_tier}"
        attempt_count["${key}"]="${state_attempts}"
        failure_count["${key}"]="${state_failures}"
        attempt_reason["${key}"]="${state_reason}"
        attempt_round["${key}"]="${state_round}"
        attempt_job_id["${key}"]="${state_job_id:-}"
    done < "${attempt_file}"
fi

# Reconciliation touches every experiment on every round, so these helpers avoid
# command substitution and any other subshell.
experiment_is_complete() {
    local classifier="$1"
    local dataset="$2"
    local resample="$3"
    local prefix="${classifier_results_abs[${classifier}]}/${classifier}/Predictions/${dataset}"

    if [[ ! -s "${prefix}/testResample${resample}.csv" ]]; then
        return 1
    fi
    if [[ -n "${classifier_train_arg[${classifier}]}" &&
          ! -s "${prefix}/trainResample${resample}.csv" ]]; then
        return 1
    fi
    return 0
}

latest_log_result=""
latest_log_for() {
    local classifier="$1"
    local dataset="$2"
    local resample="$3"
    local directory="${out_dir}/${classifier}"
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

# Locate the command list belonging to an active task-farm allocation.
active_command_file_for_job() {
    local job_id="$1"
    local job_information=""
    local stdout_path=""
    local stdout_name=""
    local stdout_directory=""
    local suffix=""
    local candidate=""

    job_information=$(scontrol show job -o "${job_id}" 2>/dev/null || true)
    if [[ "${job_information}" =~ StdOut=([^[:space:]]+) ]]; then
        stdout_path="${BASH_REMATCH[1]}"
        stdout_name="${stdout_path##*/}"
        stdout_directory="${stdout_path%/*}"

        if [[ "${stdout_name}" == "${job_id}-"* ]]; then
            suffix="${stdout_name#"${job_id}"-}"
        elif [[ "${stdout_name}" == "%A-"* ]]; then
            suffix="${stdout_name#"%A-"}"
        fi
        suffix="${suffix%.out}"
        candidate="${stdout_directory}/generatedCommandList-${suffix}.txt"
        if [[ -f "${candidate}" ]]; then
            printf '%s' "${candidate}"
        fi
    fi
}

# Read every currently running or pending node allocation from this workflow.
# Their commands must not be diagnosed as failed or submitted a second time.
declare -A active_experiment=()
declare -A active_job_name_by_id=()
declare -A active_job_state_by_id=()
declare -A active_job_node_by_id=()
declare -A active_job_memory_by_id=()
declare -A active_job_command_count=()
declare -A active_job_complete_count=()
declare -A active_job_started_count=()
declare -A active_job_waiting_count=()
active_node_job_ids=()
occupied_job_ids=()
active_job_mapping_errors=()
command_regex='classification_experiments\.py[[:space:]]+[^[:space:]]+[[:space:]]+[^[:space:]]+[[:space:]]+([^[:space:]]+)[[:space:]]+([^[:space:]]+)[[:space:]]+([0-9]+)'
redirect_regex='>[[:space:]]+([^[:space:]]+)[[:space:]]+2>&1'

while IFS='|' read -r active_job_id active_job_name active_job_state \
    active_job_node active_job_memory; do
    if [[ -z "${active_job_id}" || "${active_job_name}" == *supervisor* ||
          "${active_job_name}" == *report* ]]; then
        continue
    fi
    if [[ "${active_job_name}" != "${job_name_prefix}-r"* ]]; then
        continue
    fi
    occupied_job_ids+=("${active_job_id}")

    active_command_file=$(active_command_file_for_job "${active_job_id}")
    if [[ ! -f "${active_command_file}" ]]; then
        active_job_mapping_errors+=(
            "${active_job_id} (${active_job_name}, ${active_job_state})"
        )
        continue
    fi

    active_node_job_ids+=("${active_job_id}")
    active_job_name_by_id["${active_job_id}"]="${active_job_name}"
    active_job_state_by_id["${active_job_id}"]="${active_job_state}"
    active_job_node_by_id["${active_job_id}"]="${active_job_node}"
    active_job_memory_by_id["${active_job_id}"]="${active_job_memory}"
    active_commands=0
    active_complete=0
    active_started=0
    active_waiting=0
    while IFS= read -r active_command_line || [[ -n "${active_command_line}" ]]; do
        if [[ "${active_command_line}" =~ ${command_regex} ]]; then
            active_key="${BASH_REMATCH[1]}|${BASH_REMATCH[2]}|${BASH_REMATCH[3]}"
            active_commands=$((active_commands + 1))
            if experiment_is_complete \
                "${BASH_REMATCH[1]}" "${BASH_REMATCH[2]}" "${BASH_REMATCH[3]}"; then
                active_complete=$((active_complete + 1))
                continue
            fi

            active_experiment["${active_key}"]="${active_job_id}"
            attempt_job_id["${active_key}"]="${active_job_id}"
            active_output_log=""
            if [[ "${active_command_line}" =~ ${redirect_regex} ]]; then
                active_output_log="${BASH_REMATCH[1]}"
            fi
            if [[ -n "${active_output_log}" && -e "${active_output_log}" ]]; then
                active_started=$((active_started + 1))
            else
                active_waiting=$((active_waiting + 1))
            fi
        fi
    done < "${active_command_file}"
    active_job_command_count["${active_job_id}"]="${active_commands}"
    active_job_complete_count["${active_job_id}"]="${active_complete}"
    active_job_started_count["${active_job_id}"]="${active_started}"
    active_job_waiting_count["${active_job_id}"]="${active_waiting}"
done < <(
    squeue --noheader --user="${username}" --partition="${queue}" \
        --states=RUNNING,PENDING --format='%i|%200j|%T|%R|%m'
)

if ((${#active_job_mapping_errors[@]} > 0)); then
    echo "ERROR: active node jobs could not be mapped to their command lists:" >&2
    printf '  %s\n' "${active_job_mapping_errors[@]}" >&2
    echo "Refusing to refill because their experiments cannot safely be excluded." >&2
    exit 1
fi

available_node_slots=$((node_count - ${#occupied_job_ids[@]}))
if ((available_node_slots < 0)); then
    available_node_slots=0
fi

# Classify the most recent attempt of an experiment that has no result.
#
# OOM      a memory kill is recorded in the log
# FAILED   Python or Slurm reported some other error
# TIMEOUT  the containing allocation reached its wall-clock limit and is retried
#          without increasing the memory tier
# KILLED   a log but no result and no error, treated as memory suspect
# NOLOG    nothing ever started, usually the round ended before its turn
classify_failure_result=""
declare -A allocation_state_cache=()
allocation_state_result=""

allocation_state_for_job() {
    local job_id="$1"
    local state=""

    allocation_state_result=""
    if [[ -z "${job_id}" ]]; then
        return
    fi
    if [[ -n "${allocation_state_cache[${job_id}]+present}" ]]; then
        allocation_state_result="${allocation_state_cache[${job_id}]}"
        return
    fi
    state=$(sacct --noheader --parsable2 --jobs "${job_id}" \
        --format=JobIDRaw,State 2>/dev/null | \
        awk -F'|' -v wanted="${job_id}" '$1 == wanted { print $2; exit }' || true)
    state="${state%%+*}"
    state="${state%% *}"
    allocation_state_cache["${job_id}"]="${state}"
    allocation_state_result="${state}"
}

classify_failure() {
    local classifier="$1"
    local dataset="$2"
    local resample="$3"
    local log
    local key="${classifier}|${dataset}|${resample}"
    local job_id="${attempt_job_id[${key}]-}"
    local allocation_state=""

    latest_log_for "${classifier}" "${dataset}" "${resample}"
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

    allocation_state_for_job "${job_id}"
    allocation_state="${allocation_state_result}"
    case "${allocation_state}" in
        OUT_OF_MEMORY)
            classify_failure_result="OOM"
            return
            ;;
        TIMEOUT)
            classify_failure_result="TIMEOUT"
            return
            ;;
    esac

    classify_failure_result="KILLED"
}

# ==============================================================================
# Reconcile the previous round and choose this round's memory tiers
# ==============================================================================

max_tier=${#memory_tiers_gib[@]}

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
timeouts_observed=0

for classifier in "${classifiers[@]}"; do
    for dataset in "${datasets[@]}"; do
        for ((resample = start_fold - 1; resample < max_folds; resample++)); do
            key="${classifier}|${dataset}|${resample}"

            if experiment_is_complete "${classifier}" "${dataset}" "${resample}"; then
                completed_total=$((completed_total + 1))
                if [[ -n "${attempt_reason[${key}]+present}" ]]; then
                    attempt_reason["${key}"]="COMPLETE"
                fi
                continue
            fi

            if [[ -n "${active_experiment[${key}]+present}" ]]; then
                attempt_reason["${key}"]="SUBMITTED"
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

            # A pending tier can wait through several refill invocations. Only
            # diagnose a submitted attempt once, otherwise one failure can be
            # counted repeatedly without another execution.
            if ((attempts > 0)) && [[ "${reason}" == "SUBMITTED" ]]; then
                classify_failure "${classifier}" "${dataset}" "${resample}"
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
                    TIMEOUT)
                        failures=$((failures + 1))
                        timeouts_observed=$((timeouts_observed + 1))
                        ;;
                esac

                attempt_tier["${key}"]="${tier}"
                failure_count["${key}"]="${failures}"
                attempt_reason["${key}"]="${reason}"
            fi

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

total_experiments=$((${#classifiers[@]} * ${#datasets[@]} * (max_folds - start_fold + 1)))

echo "UCR ${workflow_label} run - round ${round} of at most ${max_rounds}"
echo "Results root:      ${results_root}"
echo "Data:              ${data_dir}"
echo "Classifiers:       ${#classifiers[@]}"
for classifier in "${classifiers[@]}"; do
    printf '  %-22s %-34s %s\n' \
        "${classifier}" \
        "${classifier_results_dir[${classifier}]}" \
        "train ${classifier_train_arg[${classifier}]:-none}"
done
echo "Datasets:          ${#datasets[@]}"
echo "Resamples:         $((max_folds - start_fold + 1))"
echo "Experiments:       ${total_experiments}"
echo "Complete:          ${completed_total}"
echo "Pending:           ${#pending_keys[@]}"
echo "Active node jobs:  ${#active_node_job_ids[@]} (${active_node_job_ids[*]-none})"
echo "Free node slots:   ${available_node_slots}/${node_count}"
echo "Active experiments: ${#active_experiment[@]}"
echo "Escalated:         ${oom_escalated} (memory kill or silent death)"
echo "Timed out:         ${timeouts_observed} (retried without raising memory)"
if [[ -n "${tier_ceiling_note}" ]]; then
    echo "Memory ceiling:    ${tier_ceiling_note}"
fi
echo "Retired:           ${#dead_keys[@]}"
if [[ -n "${estimator_source_note}" ]]; then
    echo "WARNING:           ${estimator_source_note}"
fi
echo "tsml-eval commit:  ${tsml_eval_head}"
echo "aeon commit:       ${aeon_head}"
echo

if ((${#active_node_job_ids[@]} > 0)); then
    echo "Active task-farm allocations"
    echo "----------------------------"
    printf '%-12s %-36s %-9s %-12s %-8s %8s %9s %11s %9s\n' \
        "JOBID" "NAME" "STATE" "NODE/REASON" "MEMORY" \
        "COMMANDS" "COMPLETE" "STARTED/INC" "WAITING"
    for active_job_id in "${active_node_job_ids[@]}"; do
        printf '%-12s %-36s %-9s %-12s %-8s %8d %9d %11d %9d\n' \
            "${active_job_id}" \
            "${active_job_name_by_id[${active_job_id}]}" \
            "${active_job_state_by_id[${active_job_id}]}" \
            "${active_job_node_by_id[${active_job_id}]}" \
            "${active_job_memory_by_id[${active_job_id}]}" \
            "${active_job_command_count[${active_job_id}]}" \
            "${active_job_complete_count[${active_job_id}]}" \
            "${active_job_started_count[${active_job_id}]}" \
            "${active_job_waiting_count[${active_job_id}]}"
    done
    echo
fi

if ((${#pending_keys[@]} == 0)); then
    echo "Nothing left to run."
    if ((${#dead_keys[@]} > 0)); then
        echo "Retired experiments that never produced a result:"
        printf '  %s\n' "${dead_keys[@]}"
    fi
    exit 0
fi

# ==============================================================================
# Allocate the currently free nodes across the memory tiers in this round
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

declare -A tier_slots=()
remaining_slots=${available_node_slots}
for tier in "${active_tiers[@]}"; do
    tier_slots["${tier}"]=0
done

if ((${#active_tiers[@]} <= remaining_slots)); then
    for tier in "${active_tiers[@]}"; do
        tier_slots["${tier}"]=1
        remaining_slots=$((remaining_slots - 1))
    done
fi

while ((remaining_slots > 0)); do
    best_tier=""
    best_load=-1
    for tier in "${active_tiers[@]}"; do
        assigned_slots=${tier_slots[${tier}]}
        if ((assigned_slots < 1)); then
            assigned_slots=1
        fi
        load=$((
            ${tier_pending_count[${tier}]} /
            (assigned_slots * ${tier_cpus[${tier}]})
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

run_id=$(date +%Y%m%d%H%M%S)-${SLURM_JOB_ID:-$$}
submission_dir="${results_root}/batch-submissions-completion/${run_id}-round${round}"
mkdir -p "${submission_dir}"

total_commands=0
submitted_job_ids=()

write_command() {
    local classifier="$1"
    local dataset="$2"
    local resample="$3"
    local batch_id="$4"
    local command_file="$5"
    local experiment_output
    local command_line
    local -a command

    experiment_output="${out_dir}/${classifier}/output-${dataset}-${resample}-${batch_id}.txt"

    command=(
        "${python_path}"
        -u
        "${script_file_path}"
        "${data_dir}"
        "${classifier_results_abs[${classifier}]}"
        "${classifier}"
        "${dataset}"
        "${resample}"
    )

    if [[ -n "${classifier_train_arg[${classifier}]}" ]]; then
        command+=("${classifier_train_arg[${classifier}]}")
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
    local submitted_command_line
    local submitted_key

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

current_aeon_commit=\$(git -C "${aeon_dir}" rev-parse HEAD)
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
echo "tsml-eval commit:  \$(git -C "${tsml_eval_dir}" rev-parse HEAD)"
echo "aeon commit:       \$(git -C "${aeon_dir}" rev-parse HEAD)"
echo "Command file:      ${command_file}"
echo

staskfarm "${command_file}"
SUB

    if [[ "${dry_run}" == "true" ]]; then
        echo "${batch_label}: would submit ${cmd_count} command(s) on ${cpu_count} CPU(s) at ${memory_gib} GiB each."
        return
    fi

    sbatch_output=$(sbatch "${submission_file}")
    job_id="${sbatch_output##* }"
    submitted_job_ids+=("${job_id}")
    while IFS= read -r submitted_command_line || [[ -n "${submitted_command_line}" ]]; do
        if [[ "${submitted_command_line}" =~ ${command_regex} ]]; then
            submitted_key="${BASH_REMATCH[1]}|${BASH_REMATCH[2]}|${BASH_REMATCH[3]}"
            attempt_job_id["${submitted_key}"]="${job_id}"
        fi
    done < "${command_file}"
    echo "${batch_label}: ${cmd_count} command(s) on ${cpu_count} CPU(s) at ${memory_gib} GiB each -> ${sbatch_output}"
}

# Build the per node command files. Within a tier, experiments are sorted by raw
# data size and dealt to the least loaded node, so the nodes finish close
# together instead of one node holding every large problem.
for tier in "${active_tiers[@]}"; do
    memory_gib="${memory_tiers_gib[$((tier - 1))]}"
    slots="${tier_slots[${tier}]}"

    if ((slots < 1)); then
        continue
    fi

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
        IFS='|' read -r command_classifier command_dataset command_resample <<< "${key}"

        best_slot=0
        best_load=-1
        for ((slot = 0; slot < slots; slot++)); do
            if ((best_load < 0 || slot_loads[slot] < best_load)); then
                best_load=${slot_loads[slot]}
                best_slot=${slot}
            fi
        done

        write_command \
            "${command_classifier}" "${command_dataset}" "${command_resample}" \
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
            IFS='|' read -r sort_classifier sort_dataset sort_resample <<< "${key}"
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
        IFS='|' read -r state_classifier state_dataset state_resample <<< "${key}"
        printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
            "${state_classifier}" "${state_dataset}" "${state_resample}" \
            "${attempt_tier[${key}]-1}" \
            "${attempt_count[${key}]-0}" \
            "${failure_count[${key}]-0}" \
            "${attempt_reason[${key}]-UNKNOWN}" \
            "${attempt_round[${key}]-0}" \
            "${attempt_job_id[${key}]-}" \
            >> "${tmp_attempt_file}"
    done
    mv "${tmp_attempt_file}" "${attempt_file}"
fi

# ==============================================================================
# Round summary
# ==============================================================================

send_round_summary() {
    local summary_file="${state_dir}/round-${round}-summary.txt"
    local subject
    local percent
    local mailer=""

    percent=$(awk -v done="${completed_total}" -v all="${total_experiments}" \
        'BEGIN { printf "%.1f", 100 * done / all }')

    {
        printf 'UCR %s, round %s\n' "${workflow_label}" "${round}"
        printf 'Host: %s at %s\n\n' "$(hostname)" "$(date '+%Y-%m-%d %H:%M:%S %Z')"
        printf 'Complete:   %s/%s (%s%%)\n' \
            "${completed_total}" "${total_experiments}" "${percent}"
        printf 'Pending:    %s\n' "${#pending_keys[@]}"
        printf 'Escalated:  %s (memory kill or silent death)\n' "${oom_escalated}"
        printf 'Timed out:  %s (same-memory retry)\n' "${timeouts_observed}"
        printf 'Retired:    %s\n' "${#dead_keys[@]}"
        printf 'Submitted:  %s command(s) over %s node job(s)\n\n' \
            "${total_commands}" "${#submitted_job_ids[@]}"

        if ((${#dead_keys[@]} > 0)); then
            printf 'Retired experiments:\n'
            printf '  %s\n' "${dead_keys[@]}"
            printf '\n'
        fi

        printf 'Results:     %s\n' "${results_root}"
        printf 'Submissions: %s\n' "${submission_dir}"
    } > "${summary_file}"

    if [[ "${email_updates,,}" != "true" ]]; then
        return
    fi

    subject="UCR ${workflow_label} round ${round}: ${percent}% complete, ${#pending_keys[@]} left"

    for candidate in mail mailx sendmail; do
        if command -v "${candidate}" >/dev/null 2>&1; then
            mailer="${candidate}"
            break
        fi
    done

    case "${mailer}" in
        mail|mailx)
            "${mailer}" -s "${subject}" "${mailto}" < "${summary_file}" || \
                echo "Round summary mail failed; it is saved at ${summary_file}."
            ;;
        sendmail)
            {
                printf 'To: %s\n' "${mailto}"
                printf 'Subject: %s\n\n' "${subject}"
                cat "${summary_file}"
            } | sendmail -t || \
                echo "Round summary mail failed; it is saved at ${summary_file}."
            ;;
        *)
            echo "No mail command found; summary saved at ${summary_file}."
            ;;
    esac
}

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

send_round_summary

# Continuous refill: arm one dependent successor per node job just submitted.
# When that node finishes, its successor reconciles and refills only the slot it
# freed, so the nodes stay saturated instead of idling until a whole round of
# four ends. Every running node therefore always carries exactly one successor,
# which is what keeps the pipeline live without a round barrier. Concurrent
# successors are serialized by the shared flock, and each submits only into the
# slots squeue shows free, so the four-node cap is never exceeded.
if [[ "${chain}" != "true" ]]; then
    echo "Chaining disabled: run the script again to continue."
elif ((round >= max_rounds)); then
    echo "Refill generation limit ${max_rounds} reached: no successors armed."
else
    next_round=$((round + 1))

    arm_successor() {
        local dependency="$1"
        local tag="$2"
        local supervisor_file="${submission_dir}/generatedSupervisor-r${next_round}-${tag}.sub"

        # afterany, not afterok: a node that dies is exactly the case a successor
        # must reconcile and retry at a higher memory tier.
        cat > "${supervisor_file}" <<SUP
#!/bin/bash
#SBATCH --mail-type=${supervisor_mail}
#SBATCH --mail-user=${mailto}
#SBATCH --job-name=${job_name_prefix}-supervisor-r${next_round}-${tag}
#SBATCH --partition=${queue}
#SBATCH --time=${supervisor_time}
#SBATCH --output=${submission_dir}/%A-supervisor-r${next_round}-${tag}.out
#SBATCH --error=${submission_dir}/%A-supervisor-r${next_round}-${tag}.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=${supervisor_memory}
#SBATCH --dependency=${dependency}

. /etc/profile
set -e

bash "${script_path}" \\
    --round ${next_round} \\
    --max-rounds ${max_rounds} \\
    --dataset-list "${dataset_list_file}"
SUP

        local output
        output=$(sbatch "${supervisor_file}")
        echo "  successor ${tag} on ${dependency}: ${output}"
    }

    armed=0
    for node_job_id in "${submitted_job_ids[@]}"; do
        arm_successor "afterany:${node_job_id}" "n${armed}"
        armed=$((armed + 1))
    done

    # Liveness safety only for a manual (re)launch that found every node busy and
    # so submitted nothing: keep one successor alive so pending work is not
    # stranded. Steady-state refills never need it, because each running node
    # already carries its own successor, so it is gated to the initial launch.
    if ((armed == 0 && round == 1 &&
          ${#pending_keys[@]} > 0 && ${#occupied_job_ids[@]} > 0)); then
        arm_successor "afterany:${occupied_job_ids[0]}" "wait"
        armed=1
    fi

    if ((armed > 0)); then
        echo "Armed ${armed} successor(s), one per node job; each refills its slot when its node ends."
    else
        echo "No successors armed (nothing submitted, nothing pending to wait on)."
    fi
fi

echo
echo "Results:     ${results_root}"
echo "Submissions: ${submission_dir}"
echo "State:       ${state_dir}  (per round summaries + attempts.tsv)"
