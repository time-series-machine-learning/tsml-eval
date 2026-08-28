#!/bin/bash

set -uo pipefail

# Monitor the interval-based regressor run over the 63 problem TSER archive.
#
# This script is read-only. Run it on Iridis for one snapshot:
#
#   bash monitor_tser_interval_regressors.sh
#
# Include every incomplete regressor/problem pair:
#
#   bash monitor_tser_interval_regressors.sh --details
#
# Refresh every 60 seconds (Ctrl-C to stop):
#
#   bash monitor_tser_interval_regressors.sh --watch 60
#
# Completion is counted from the result files, so it is correct whether or not
# the chain of rounds submitted by run_tser_interval_regressors.sh is still
# alive. Failure attribution comes from that script's attempt state, which is
# where the memory tier of each experiment is recorded.

username="ajb2u23"
local_path="/iridisfs/home/${username}"
results_root="${TSER_INTERVAL_RESULTS_ROOT:-${local_path}/Results/TSER/IntervalBased}"
out_dir="${results_root}/output"
state_dir="${results_root}/.tser-interval-state"
attempt_file="${state_dir}/attempts.tsv"
job_name_prefix="tser-interval"
workflow_label="interval regressors"

dataset_list_file="${TSER_INTERVAL_DATASET_LIST:-${local_path}/Code/tsml-eval/_tsml_research_resources/dataset_lists/Regression63-MonashExtendedClean.txt}"

resamples=30
memory_tiers_gib=(4 8 16 32 64 128 256 620)

details="false"
summary_only="false"
watch_seconds=0
profile="interval"

# Must match run_tser_interval_regressors.sh.
regressors=(
    "tsf-500"
    "rise-500"
    "cif-500"
    "drcif-500"
    "randomintervals-500"
    "summary-intervals"
    "quant"
    "pulsar"
)

declare -A regressor_category=()

declare -A complete_count=()
declare -A live_state=()
declare -A live_job=()
declare -A live_node=()
declare -a relevant_job_records=()
declare -a datasets=()

usage() {
    printf '%s\n' \
        "Usage:" \
        "  monitor_tser_interval_regressors.sh [--profile NAME] [--details] [--watch SECONDS]" \
        "" \
        "Options:" \
        "  --profile NAME   interval or remaining-aeon (default interval)." \
        "  --details        List every incomplete regressor/problem pair." \
        "  --summary        Per regressor progress only, for mailing." \
        "  --watch SECONDS  Refresh continuously at the specified interval." \
        "  -h, --help       Show this help."
}

while (($# > 0)); do
    case "$1" in
        --profile)
            if (($# < 2)); then
                echo "ERROR: --profile requires a value." >&2
                exit 2
            fi
            profile="$2"
            shift 2
            ;;
        --details)
            details="true"
            shift
            ;;
        --summary)
            summary_only="true"
            shift
            ;;
        --watch)
            if (($# < 2)); then
                echo "ERROR: --watch requires a refresh interval in seconds." >&2
                exit 2
            fi
            watch_seconds="$2"
            shift 2
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

case "${profile}" in
    interval)
        ;;
    remaining-aeon)
        regressors=(
            "rocket"
            "minirocket"
            "multirocket"
            "hydra"
            "multirocket-hydra"
            "1nn-dtw"
            "summary-500"
            "catch22-500"
            "freshprince-500"
            "tsfresh"
            "rist"
            "rdst"
            "dummy"
            "rotationforest"
        )
        regressor_category=(
            [rocket]="ConvolutionBased"
            [minirocket]="ConvolutionBased"
            [multirocket]="ConvolutionBased"
            [hydra]="ConvolutionBased"
            [multirocket-hydra]="ConvolutionBased"
            [1nn-dtw]="DistanceBased"
            [summary-500]="FeatureBased"
            [catch22-500]="FeatureBased"
            [freshprince-500]="FeatureBased"
            [tsfresh]="FeatureBased"
            [rist]="Hybrid"
            [rdst]="ShapeletBased"
            [dummy]="Other"
            [rotationforest]="VectorBased"
        )
        results_root="${TSER_AEON_RESULTS_ROOT:-${local_path}/Results/TSER}"
        out_dir="${TSER_AEON_OUTPUT_DIR:-${results_root}/.tser-aeon-output}"
        state_dir="${TSER_AEON_STATE_DIR:-${results_root}/.tser-aeon-state}"
        attempt_file="${state_dir}/attempts.tsv"
        job_name_prefix="tser-aeon"
        workflow_label="remaining non-deep aeon regressors"
        ;;
    *)
        echo "ERROR: --profile must be interval or remaining-aeon." >&2
        exit 2
        ;;
esac

regressor_results_root_result=""
regressor_results_root() {
    local regressor="$1"
    local category="${regressor_category[${regressor}]-}"

    if [[ -n "${category}" ]]; then
        regressor_results_root_result="${results_root}/${category}"
    else
        regressor_results_root_result="${results_root}"
    fi
}

if ! [[ "${watch_seconds}" =~ ^[0-9]+$ ]]; then
    echo "ERROR: --watch must be a non-negative integer." >&2
    exit 2
fi
if ((watch_seconds > 0 && watch_seconds < 5)); then
    echo "ERROR: use a refresh interval of at least 5 seconds." >&2
    exit 2
fi

read_datasets() {
    local line

    datasets=()
    if [[ ! -f "${dataset_list_file}" ]]; then
        return 1
    fi

    while IFS= read -r line || [[ -n "${line}" ]]; do
        line="${line//$'\r'/}"
        line="${line#"${line%%[![:space:]]*}"}"
        line="${line%"${line##*[![:space:]]}"}"
        if [[ -z "${line}" || "${line:0:1}" == "#" ]]; then
            continue
        fi
        datasets+=("${line}")
    done < "${dataset_list_file}"

    ((${#datasets[@]} > 0))
}

# One find per regressor rather than one stat per experiment. Zero length
# results are excluded, so a truncated write is not counted as complete.
collect_complete_counts() {
    local regressor
    local predictions_dir
    local path
    local relative
    local dataset
    local file
    local index

    complete_count=()

    for regressor in "${regressors[@]}"; do
        regressor_results_root "${regressor}"
        predictions_dir="${regressor_results_root_result}/${regressor}/Predictions"
        if [[ ! -d "${predictions_dir}" ]]; then
            continue
        fi

        while IFS= read -r path; do
            relative="${path#"${predictions_dir}/"}"
            dataset="${relative%%/*}"
            file="${relative##*/}"
            index="${file#testResample}"
            index="${index%.csv}"

            if ! [[ "${index}" =~ ^[0-9]+$ ]] || ((index >= resamples)); then
                continue
            fi

            complete_count["${regressor}|${dataset}"]=$((
                ${complete_count["${regressor}|${dataset}"]-0} + 1
            ))
        done < <(
            find "${predictions_dir}" \
                -mindepth 2 -maxdepth 2 \
                -name 'testResample*.csv' \
                -size +0c \
                2>/dev/null
        )
    done
}

command_file_for_job() {
    local job_id="$1"
    local job_information=""
    local stdout_path=""
    local stdout_name=""
    local stdout_directory=""
    local suffix=""
    local candidate=""

    # The submission script writes StdOut as <jobid>-<batch id>.out beside
    # generatedCommandList-<batch id>.txt, so the command file can be found for
    # pending jobs as well as running ones.
    if command -v scontrol >/dev/null 2>&1; then
        job_information=$(scontrol show job -o "${job_id}" 2>/dev/null)
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
    fi
}

refresh_slurm_activity() {
    local job_id
    local partition
    local job_name
    local job_state
    local elapsed
    local nodes
    local reason
    local command_file
    local command_line
    local regressor
    local dataset
    local resample
    local output_log
    local result_file
    local combo_state
    local key
    local command_regex='regression_experiments\.py[[:space:]]+[^[:space:]]+[[:space:]]+[^[:space:]]+[[:space:]]+([^[:space:]]+)[[:space:]]+([^[:space:]]+)[[:space:]]+([0-9]+)'
    local redirect_regex='>[[:space:]]+([^[:space:]]+)[[:space:]]+2>&1'

    live_state=()
    live_job=()
    live_node=()
    relevant_job_records=()

    if ! command -v squeue >/dev/null 2>&1; then
        return
    fi

    while IFS='|' read -r \
        job_id partition job_name job_state elapsed nodes reason; do
        if [[ -z "${job_id}" ]]; then
            continue
        fi
        if [[ "${job_name}" != "${job_name_prefix}"* ]]; then
            continue
        fi

        relevant_job_records+=(
            "${job_id}|${partition}|${job_name}|${job_state}|${elapsed}|${nodes}|${reason}"
        )

        # The supervisor carries no experiments of its own.
        if [[ "${job_name,,}" == *supervisor* ]]; then
            continue
        fi

        command_file=$(command_file_for_job "${job_id}")
        if [[ ! -f "${command_file}" ]]; then
            continue
        fi

        while IFS= read -r command_line || [[ -n "${command_line}" ]]; do
            if [[ ! "${command_line}" =~ ${command_regex} ]]; then
                continue
            fi
            regressor="${BASH_REMATCH[1]}"
            dataset="${BASH_REMATCH[2]}"
            resample="${BASH_REMATCH[3]}"

            regressor_results_root "${regressor}"
            result_file="${regressor_results_root_result}/${regressor}/Predictions/${dataset}/testResample${resample}.csv"
            if [[ -s "${result_file}" ]]; then
                # The task-farm allocation may remain alive after this command
                # finished while other commands continue. Do not report a
                # completed experiment as still running merely because its log
                # exists.
                continue
            fi

            output_log=""
            if [[ "${command_line}" =~ ${redirect_regex} ]]; then
                output_log="${BASH_REMATCH[1]}"
            fi

            case "${job_state}" in
                RUNNING)
                    # Shell redirection creates the per-experiment log the
                    # moment staskfarm starts that command. No log means it is
                    # still waiting for a free task slot in the allocation.
                    if [[ -n "${output_log}" && -e "${output_log}" ]]; then
                        combo_state="RUNNING"
                    else
                        combo_state="QUEUED"
                    fi
                    ;;
                PENDING|CONFIGURING)
                    combo_state="PENDING"
                    ;;
                *)
                    continue
                    ;;
            esac

            key="${regressor}|${dataset}|${resample}"
            if [[ "${live_state[${key}]-}" == "RUNNING" ]]; then
                continue
            fi
            if [[ "${live_state[${key}]-}" == "QUEUED" && "${combo_state}" == "PENDING" ]]; then
                continue
            fi
            live_state["${key}"]="${combo_state}"
            live_job["${key}"]="${job_id}"
            if [[ "${job_state}" == "RUNNING" ]]; then
                live_node["${key}"]="${reason}"
            else
                live_node["${key}"]="-"
            fi
        done < "${command_file}"
    done < <(
        squeue \
            --noheader \
            --user="${username}" \
            --format="%i|%P|%j|%T|%M|%D|%R" \
            2>/dev/null
    )
}

print_running_nodes() {
    local record
    local job_id
    local partition
    local job_name
    local job_state
    local elapsed
    local nodes
    local reason
    local node
    local node_key
    local current_count
    local -A allocation_count=()

    for record in "${relevant_job_records[@]}"; do
        IFS='|' read -r \
            job_id partition job_name job_state elapsed nodes reason <<< "${record}"
        if [[ "${job_state}" != "RUNNING" || "${job_name,,}" == *supervisor* || \
              "${job_name,,}" == *report* ]]; then
            continue
        fi
        node_key="${reason:-unknown}"
        current_count=${allocation_count["${node_key}"]-0}
        allocation_count["${node_key}"]=$((current_count + 1))
    done

    if ((${#allocation_count[@]} == 0)); then
        echo "Running nodes: none"
        return
    fi

    echo "Running nodes:"
    while IFS= read -r node; do
        printf '  %s (%d allocation(s))\n' \
            "${node}" "${allocation_count["${node}"]}"
    done < <(printf '%s\n' "${!allocation_count[@]}" | sort)
}

print_relevant_queue() {
    local record
    local job_id
    local partition
    local job_name
    local job_state
    local elapsed
    local nodes
    local reason

    echo "Relevant Slurm jobs"
    echo "-------------------"

    if ! command -v squeue >/dev/null 2>&1; then
        echo "squeue is unavailable in this shell."
        echo
        return
    fi

    if ((${#relevant_job_records[@]} == 0)); then
        echo "No matching running or pending jobs."
        echo "If work is outstanding the chain has stopped; restart it with"
        echo "  bash run_tser_interval_regressors.sh --profile ${profile}"
        echo
        return
    fi

    printf '%-18s %-9s %-34s %-11s %-10s %-6s %s\n' \
        "JOBID" "PARTITION" "NAME" "STATE" "TIME" "NODES" "NODELIST(REASON)"
    for record in "${relevant_job_records[@]}"; do
        IFS='|' read -r \
            job_id partition job_name job_state elapsed nodes reason <<< "${record}"
        printf '%-18s %-9s %-34s %-11s %-10s %-6s %s\n' \
            "${job_id}" "${partition}" "${job_name}" "${job_state}" \
            "${elapsed}" "${nodes}" "${reason}"
    done
    echo
}

print_current_activity() {
    local key
    local regressor
    local dataset
    local resample
    local running_count=0
    local queued_count=0
    local pending_count=0
    local -a running_records=()

    for key in "${!live_state[@]}"; do
        case "${live_state[${key}]}" in
            RUNNING)
                running_count=$((running_count + 1))
                IFS='|' read -r regressor dataset resample <<< "${key}"
                running_records+=(
                    "${regressor}|${dataset}|${resample}|${live_job[${key}]}|${live_node[${key}]}"
                )
                ;;
            QUEUED) queued_count=$((queued_count + 1)) ;;
            PENDING) pending_count=$((pending_count + 1)) ;;
        esac
    done

    echo "Current experiment activity - ${running_count} running"
    echo "----------------------------------------"
    printf '%-22s %-30s %-9s %-12s %s\n' \
        "REGRESSOR" "DATASET" "RESAMPLE" "JOBID" "NODE"

    for key in "${running_records[@]}"; do
        IFS='|' read -r regressor dataset resample job_id node <<< "${key}"
        printf '%-22s %-30s %-9s %-12s %s\n' \
            "${regressor}" "${dataset}" "${resample}" "${job_id}" "${node}"
    done

    if ((running_count == 0)); then
        echo "No experiment is currently executing."
    fi
    echo "Waiting for a task slot inside a running allocation: ${queued_count}"
    echo "Held in pending allocations:                         ${pending_count}"
    echo
}

# Failure attribution is read from the runner's attempt state instead of
# grepping 15120 logs. The runner writes it once per round.
print_attempt_state() {
    local regressor
    local dataset
    local resample
    local tier
    local attempts
    local failures
    local reason
    local last_round
    local last_job_id
    local memory
    local latest_round=0
    local -A tier_counts=()
    local -A reason_counts=()
    local -a dead_records=()

    echo "Attempt state"
    echo "-------------"

    if [[ ! -f "${attempt_file}" ]]; then
        echo "No attempt state yet: ${attempt_file}"
        echo
        return
    fi

    while IFS=$'\t' read -r regressor dataset resample tier attempts \
        failures reason last_round last_job_id; do
        if [[ -z "${regressor:-}" ]]; then
            continue
        fi
        if [[ "${last_round}" =~ ^[0-9]+$ ]] && ((last_round > latest_round)); then
            latest_round="${last_round}"
        fi
        if [[ "${reason}" == "COMPLETE" ]]; then
            continue
        fi
        reason_counts["${reason}"]=$(( ${reason_counts["${reason}"]-0} + 1 ))
        if [[ "${reason}" != "SUBMITTED" ]]; then
            continue
        fi
        tier_counts["${tier}"]=$(( ${tier_counts["${tier}"]-0} + 1 ))
    done < "${attempt_file}"

    echo "Latest submission round: ${latest_round}"

    echo "In flight by memory tier:"
    for tier in "${!tier_counts[@]}"; do
        memory="${memory_tiers_gib[$((tier - 1))]-unknown}"
        printf '  tier %s (%s GiB): %s\n' \
            "${tier}" "${memory}" "${tier_counts[${tier}]}"
    done
    if ((${#tier_counts[@]} == 0)); then
        echo "  none"
    fi

    echo "Outcomes recorded for incomplete experiments:"
    for reason in "${!reason_counts[@]}"; do
        printf '  %-10s %s\n' "${reason}" "${reason_counts[${reason}]}"
    done
    if ((${#reason_counts[@]} == 0)); then
        echo "  none"
    fi
    echo "  OOM and KILLED are retried at the next memory tier."
    echo "  TIMEOUT is retried at the same memory tier."
    echo "  DEAD experiments have exhausted their attempts and are abandoned."
    echo
}

scan_once() {
    local regressor
    local dataset
    local key
    local done_count
    local regressor_complete
    local regressor_datasets_done
    local total_complete=0
    local total_expected
    local expected_per_regressor
    local -a incomplete_details=()

    if [[ ! -d "${results_root}" ]]; then
        echo "ERROR: results directory not found:"
        echo "  ${results_root}"
        return 1
    fi

    if ! read_datasets; then
        echo "ERROR: no datasets read from:"
        echo "  ${dataset_list_file}"
        return 1
    fi

    expected_per_regressor=$((${#datasets[@]} * resamples))
    total_expected=$((expected_per_regressor * ${#regressors[@]}))

    collect_complete_counts
    refresh_slurm_activity

    printf 'TSER %s monitor - %s\n' "${workflow_label}" "$(date '+%Y-%m-%d %H:%M:%S %Z')"
    printf 'Machine: %s\n' "$(hostname -f 2>/dev/null || hostname)"
    echo "Results: ${results_root}"
    echo "Scope:   ${#regressors[@]} regressors x ${#datasets[@]} datasets x ${resamples} resamples"
    print_running_nodes
    echo

    if [[ "${summary_only}" == "true" ]]; then
        # Email summaries still need enough scheduler context to identify the
        # allocation, state and machine without logging into Iridis.
        print_relevant_queue
    else
        print_relevant_queue
        print_current_activity
        print_attempt_state
    fi

    # Live counts per regressor, from the same Slurm scan as the activity
    # table, so running and pending are attributed to the estimator rather
    # than reported only as one total.
    local live_key
    local live_regressor
    declare -A running_by_regressor=()
    declare -A pending_by_regressor=()

    for live_key in "${!live_state[@]}"; do
        live_regressor="${live_key%%|*}"
        case "${live_state[${live_key}]}" in
            RUNNING)
                running_by_regressor["${live_regressor}"]=$((
                    ${running_by_regressor[${live_regressor}]-0} + 1
                ))
                ;;
            QUEUED|PENDING)
                pending_by_regressor["${live_regressor}"]=$((
                    ${pending_by_regressor[${live_regressor}]-0} + 1
                ))
                ;;
        esac
    done

    printf '%-22s %14s %8s %8s %8s %8s\n' \
        "REGRESSOR" "COMPLETE" "PERCENT" "RUNNING" "PENDING" "TODO"
    printf '%-22s %14s %8s %8s %8s %8s\n' \
        "----------------------" "--------------" "--------" "--------" \
        "--------" "--------"

    for regressor in "${regressors[@]}"; do
        regressor_complete=0
        regressor_datasets_done=0

        for dataset in "${datasets[@]}"; do
            key="${regressor}|${dataset}"
            done_count=${complete_count[${key}]-0}
            regressor_complete=$((regressor_complete + done_count))
            if ((done_count >= resamples)); then
                regressor_datasets_done=$((regressor_datasets_done + 1))
            elif [[ "${details}" == "true" ]]; then
                incomplete_details+=("${regressor}|${dataset}|${done_count}")
            fi
        done

        total_complete=$((total_complete + regressor_complete))

        printf '%-22s %6d/%-7d %7.1f%% %8d %8d %8d\n' \
            "${regressor}" \
            "${regressor_complete}" "${expected_per_regressor}" \
            "$(awk -v done="${regressor_complete}" -v all="${expected_per_regressor}" \
                'BEGIN { printf 100 * done / all }')" \
            "${running_by_regressor[${regressor}]-0}" \
            "${pending_by_regressor[${regressor}]-0}" \
            "$((expected_per_regressor - regressor_complete))"
    done

    echo
    printf 'Overall complete: %d/%d (%.1f%%)\n' \
        "${total_complete}" \
        "${total_expected}" \
        "$(awk -v done="${total_complete}" -v all="${total_expected}" \
            'BEGIN { printf 100 * done / all }')"

    if [[ "${summary_only}" == "true" ]]; then
        echo
        echo "Use --details on the login node for the incomplete list."
        return 0
    fi

    if [[ "${details}" == "true" ]]; then
        echo
        echo "Incomplete regressor/problem pairs"
        echo "----------------------------------"
        printf '%-22s %-30s %s\n' "REGRESSOR" "DATASET" "RESAMPLES DONE"
        for key in "${incomplete_details[@]}"; do
            IFS='|' read -r regressor dataset done_count <<< "${key}"
            printf '%-22s %-30s %d/%d\n' \
                "${regressor}" "${dataset}" "${done_count}" "${resamples}"
        done
        if ((${#incomplete_details[@]} == 0)); then
            echo "None: every problem has all ${resamples} resamples."
        fi
    else
        echo
        echo "Use --details to list individual incomplete problems."
    fi
}

while true; do
    if ((watch_seconds > 0)) && [[ -t 1 ]]; then
        clear
    fi

    scan_once
    scan_status=$?

    if ((watch_seconds == 0)); then
        exit "${scan_status}"
    fi

    echo
    echo "Refreshing in ${watch_seconds} seconds; press Ctrl-C to stop."
    sleep "${watch_seconds}"
done
