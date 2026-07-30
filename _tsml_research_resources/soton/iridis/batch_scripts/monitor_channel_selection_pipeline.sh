#!/bin/bash

set -uo pipefail

# Monitor the established 25-problem EEG transform x classifier experiments.
#
# This script is read-only. Run it on Iridis for one snapshot:
#
#   bash monitor_channel_selection_pipeline.sh
#
# Include every incomplete classifier/problem pair:
#
#   bash monitor_channel_selection_pipeline.sh --details
#
# Refresh every 60 seconds (Ctrl-C to stop):
#
#   bash monitor_channel_selection_pipeline.sh --watch 60
#
# MatchWords is intentionally excluded: it is a separate case study.

username="ajb2u23"
local_path="/iridisfs/home/${username}"
results_root="${CHANNEL_SELECTION_RESULTS_ROOT:-${local_path}/Results/ChannelSelectionPipeline}"
resample=0

details="false"
watch_seconds=0

# These are the 25 EEG archive problems used in the paper. Keeping the list
# here makes the monitor independent of the submission-time batch files.
datasets=(
    "EyesOpenShut"
    "FingerMovements"
    "HandMovementDirection"
    "ButtonPress"
    "LowCost"
    "FeedbackButton"
    "MindReading"
    "FibroLiverpool"
    "FibroUEA"
    "PhotoStimulation"
    "MotorImagery"
    "PronouncedSpeech"
    "InnerSpeech"
    "VisualSpeech"
    "Alzheimers"
    "FaceDetection"
    "OpenCloseFist"
    "ImaginedFeetHands"
    "ImaginedOpenCloseFist"
    "FeetHands"
    "SongFamiliarity"
    "SitStand"
    "ShortIntervalTask"
    "MatchingPennies"
    "LongIntervalTask"
)

# Include the transform families retained for the paper summary. GMARv2 and
# GMARv4 are deliberately excluded. GMARv3 is retained for comparison with
# GMARv5, which replaces its TSelect stage with DetachRocket.
transforms=(
    "CSP"
    "ECS"
    "ECP"
    "TSelect"
    "Random"
    "Riemannian"
    "DetachRocket"
    "CaseTimeReducer"
    "CLeVerRank"
    "CLeVerCluster"
    "CLeVerHybrid"
    "GMARv3"
    "GMARv5"
)

# HC2 and the four classifiers from which it is built.
classifiers=(
    "HC2"
    "Arsenal"
    "DrCIF"
    "STC"
    "TDE"
)

# Refreshed from Slurm before each scan. Keys are "pipeline|dataset".
declare -A slurm_combo_state=()
declare -A slurm_combo_job=()
declare -A slurm_combo_log=()
declare -a relevant_job_records=()
latest_log_result=""
status_result=""

usage() {
    printf '%s\n' \
        "Usage:" \
        "  monitor_channel_selection_pipeline.sh [--details] [--watch SECONDS]" \
        "" \
        "Options:" \
        "  --details        List every incomplete classifier/problem pair." \
        "  --watch SECONDS  Refresh continuously at the specified interval." \
        "  -h, --help       Show this help."
}

while (($# > 0)); do
    case "$1" in
        --details)
            details="true"
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

if ! [[ "${watch_seconds}" =~ ^[0-9]+$ ]]; then
    echo "ERROR: --watch must be a non-negative integer." >&2
    exit 2
fi
if ((watch_seconds > 0 && watch_seconds < 5)); then
    echo "ERROR: use a refresh interval of at least 5 seconds." >&2
    exit 2
fi

latest_log_for() {
    local pipeline="$1"
    local dataset="$2"
    local output_directory="${results_root}/output/${pipeline}"
    local latest=""
    local candidate
    local -a candidates

    latest_log_result=""
    if [[ ! -d "${output_directory}" ]]; then
        return
    fi

    shopt -s nullglob
    candidates=(
        "${output_directory}"/output-"${dataset}"-"${resample}"-*.txt
    )
    shopt -u nullglob

    for candidate in "${candidates[@]}"; do
        if [[ -z "${latest}" || "${candidate}" -nt "${latest}" ]]; then
            latest="${candidate}"
        fi
    done

    latest_log_result="${latest}"
}

command_file_for_job() {
    local job_id="$1"
    local job_information=""
    local stdout_path=""
    local stdout_name=""
    local stdout_directory=""
    local suffix=""
    local candidate=""
    local master_output=""
    local line=""
    local command_file=""

    # This works for pending jobs as well as running jobs. Slurm's StdOut path
    # contains the same batch identifier used by generatedCommandList.
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
                return
            fi
        fi
    fi

    # Once a job starts, its main output explicitly records the command file.
    # Use this as a fallback for submission scripts with a different filename
    # convention.
    while IFS= read -r candidate; do
        master_output="${candidate}"
        break
    done < <(
        find "${results_root}/batch-submissions" \
            -type f \
            -name "${job_id}-*.out" \
            -print \
            2>/dev/null
    )

    if [[ -n "${master_output}" ]]; then
        while IFS= read -r line; do
            if [[ "${line}" == "Command file:"* ]]; then
                command_file="${line#Command file:}"
                command_file="${command_file#"${command_file%%[![:space:]]*}"}"
            fi
        done < "${master_output}"
    fi

    if [[ -f "${command_file}" ]]; then
        printf '%s' "${command_file}"
    fi
}

record_slurm_combo() {
    local key="$1"
    local state="$2"
    local job_id="$3"
    local output_log="$4"
    local existing="${slurm_combo_state[${key}]-}"

    # Prefer a started task over one waiting within a task farm, and prefer
    # either over a duplicate pending retry.
    if [[ "${existing}" == "RUNNING" ]]; then
        return
    fi
    if [[ "${existing}" == "QUEUED" && "${state}" == "PENDING" ]]; then
        return
    fi

    slurm_combo_state["${key}"]="${state}"
    slurm_combo_job["${key}"]="${job_id}"
    slurm_combo_log["${key}"]="${output_log}"
}

refresh_slurm_activity() {
    local job_id
    local partition
    local job_name
    local job_state
    local elapsed
    local nodes
    local reason
    local lower_name
    local command_file
    local command_line
    local pipeline
    local dataset
    local command_resample
    local output_log
    local combo_state
    local key
    local command_regex='classification_experiments\.py[[:space:]]+[^[:space:]]+[[:space:]]+[^[:space:]]+[[:space:]]+([^[:space:]]+)[[:space:]]+([^[:space:]]+)[[:space:]]+([0-9]+)'
    local redirect_regex='>[[:space:]]+([^[:space:]]+)[[:space:]]+2>&1'

    slurm_combo_state=()
    slurm_combo_job=()
    slurm_combo_log=()
    relevant_job_records=()

    if ! command -v squeue >/dev/null 2>&1; then
        return
    fi

    while IFS='|' read -r \
        job_id partition job_name job_state elapsed nodes reason; do
        [[ -z "${job_id}" ]] && continue
        lower_name="${job_name,,}"
        # This explicitly includes task farms such as eeg-gmarv3-batch3. The
        # command file is inspected below, so every outstanding experiment in
        # the allocation is mapped to its pipeline/problem pair.
        if [[ "${lower_name}" != *eeg* \
            && "${lower_name}" != *channel* \
            && "${lower_name}" != *gmar* ]]; then
            continue
        fi

        relevant_job_records+=(
            "${job_id}|${partition}|${job_name}|${job_state}|${elapsed}|${nodes}|${reason}"
        )

        command_file=$(command_file_for_job "${job_id}")
        if [[ ! -f "${command_file}" ]]; then
            continue
        fi

        while IFS= read -r command_line || [[ -n "${command_line}" ]]; do
            if [[ ! "${command_line}" =~ ${command_regex} ]]; then
                continue
            fi
            pipeline="${BASH_REMATCH[1]}"
            dataset="${BASH_REMATCH[2]}"
            command_resample="${BASH_REMATCH[3]}"
            if [[ "${command_resample}" != "${resample}" ]]; then
                continue
            fi

            output_log=""
            if [[ "${command_line}" =~ ${redirect_regex} ]]; then
                output_log="${BASH_REMATCH[1]}"
            fi

            case "${job_state}" in
                RUNNING)
                    # Shell redirection creates the per-experiment log when
                    # staskfarm starts that command. No log means it is still
                    # waiting for a free task slot inside the allocation.
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

            key="${pipeline}|${dataset}"
            record_slurm_combo \
                "${key}" "${combo_state}" "${job_id}" "${output_log}"
        done < "${command_file}"
    done < <(
        squeue \
            --noheader \
            --user="${username}" \
            --format="%i|%P|%j|%T|%M|%D|%R" \
            2>/dev/null
    )
}

status_for() {
    local pipeline="$1"
    local dataset="$2"
    local result_file
    local latest_log
    local key="${pipeline}|${dataset}"
    local live_state="${slurm_combo_state[${key}]-}"
    local live_log="${slurm_combo_log[${key}]-}"

    status_result=""
    result_file="${results_root}/${pipeline}/Predictions/${dataset}/testResample${resample}.csv"

    if [[ -s "${result_file}" ]]; then
        status_result="COMPLETE"
        return
    fi

    if [[ -n "${live_log}" && -s "${live_log}" ]]; then
        if grep -Eiq \
            'out[ -]?of[ -]?memory|OUT_OF_MEMORY|oom[_-]kill|Killed process|MemoryError|Cannot allocate memory|std::bad_alloc' \
            "${live_log}"; then
            status_result="OOM"
            return
        fi
        if grep -Eiq \
            'Traceback \(most recent call last\)|Segmentation fault|^ERROR:|slurmstepd: error:|Exception:' \
            "${live_log}"; then
            status_result="FAILED"
            return
        fi
    fi

    case "${live_state}" in
        RUNNING)
            status_result="RUNNING"
            return
            ;;
        QUEUED)
            status_result="QUEUED"
            return
            ;;
        PENDING)
            status_result="PENDING"
            return
            ;;
    esac

    if [[ -e "${result_file}" ]]; then
        status_result="EMPTY"
        return
    fi

    latest_log_for "${pipeline}" "${dataset}"
    latest_log="${latest_log_result}"
    if [[ -z "${latest_log}" ]]; then
        status_result="NOTSTARTED"
        return
    fi
    if [[ ! -s "${latest_log}" ]]; then
        status_result="WAITING"
        return
    fi

    if grep -Eiq \
        'out[ -]?of[ -]?memory|OUT_OF_MEMORY|oom[_-]kill|Killed process|MemoryError|Cannot allocate memory|std::bad_alloc' \
        "${latest_log}"; then
        status_result="OOM"
        return
    fi
    if grep -Eiq \
        'Traceback \(most recent call last\)|Segmentation fault|^ERROR:|slurmstepd: error:|Exception:' \
        "${latest_log}"; then
        status_result="FAILED"
        return
    fi

    # A non-empty output log exists, but a result has not appeared and the log
    # has no recognised failure. It may be running, queued for a retry, or have
    # stopped without a conventional Python/Slurm error.
    status_result="LOGGED"
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
        echo
        return
    fi

    printf '%-18s %-9s %-32s %-11s %-10s %-6s %s\n' \
        "JOBID" "PARTITION" "NAME" "STATE" "TIME" "NODES" "NODELIST(REASON)"
    for record in "${relevant_job_records[@]}"; do
        IFS='|' read -r \
            job_id partition job_name job_state elapsed nodes reason <<< "${record}"
        printf '%-18s %-9s %-32s %-11s %-10s %-6s %s\n' \
            "${job_id}" "${partition}" "${job_name}" "${job_state}" \
            "${elapsed}" "${nodes}" "${reason}"
    done
    echo
}

pipeline_is_monitored() {
    local candidate="$1"
    local transform
    local classifier

    for classifier in "${classifiers[@]}"; do
        for transform in "${transforms[@]}"; do
            if [[ "${candidate}" == "${transform}-${classifier}" ]]; then
                return 0
            fi
        done
    done

    return 1
}

print_current_activity() {
    local key
    local pipeline
    local dataset
    local state
    local job_id
    local status
    local record
    local running_count=0
    local displayed_count=0
    local -a activity_records=()

    for key in "${!slurm_combo_state[@]}"; do
        IFS='|' read -r pipeline dataset <<< "${key}"
        # Use the same scope as the summary table. Active jobs can contain
        # historical variants such as GMARv2 or GMARv4, but those variants are
        # deliberately excluded from the monitored paper results.
        if ! pipeline_is_monitored "${pipeline}"; then
            continue
        fi
        status_for "${pipeline}" "${dataset}"
        status="${status_result}"
        case "${status}" in
            RUNNING|QUEUED|PENDING)
                state="${status}"
                job_id="${slurm_combo_job[${key}]}"
                activity_records+=(
                    "${state}|${pipeline}|${dataset}|${job_id}"
                )
                displayed_count=$((displayed_count + 1))
                if [[ "${state}" == "RUNNING" ]]; then
                    running_count=$((running_count + 1))
                fi
                ;;
        esac
    done

    echo "Current experiment activity - ${running_count}"
    echo "--------------------------------"
    printf '%-10s %-27s %-28s %s\n' "STATE" "PIPELINE" "DATASET" "JOBID"

    for record in "${activity_records[@]}"; do
        IFS='|' read -r state pipeline dataset job_id <<< "${record}"
        printf '%-10s %-27s %-28s %s\n' \
            "${state}" "${pipeline}" "${dataset}" "${job_id}"
    done

    if ((displayed_count == 0)); then
        echo "No incomplete experiments could be mapped to an active Slurm job."
    fi
    echo
}

scan_once() {
    local transform
    local classifier
    local pipeline
    local dataset
    local status
    local complete
    local running
    local queued
    local pending
    local empty
    local oom
    local failed
    local logged
    local waiting
    local not_started
    local expected_per_pipeline=${#datasets[@]}
    local total_pipelines=$((${#transforms[@]} * ${#classifiers[@]}))
    local total_expected=$((total_pipelines * expected_per_pipeline))
    local total_complete=0
    local total_running=0
    local total_queued=0
    local total_pending=0
    local total_empty=0
    local total_oom=0
    local total_failed=0
    local total_logged=0
    local total_waiting=0
    local total_not_started=0
    local detail_key
    local detail_log
    local -a incomplete_details=()

    if [[ ! -d "${results_root}" ]]; then
        echo "ERROR: results directory not found:"
        echo "  ${results_root}"
        return 1
    fi

    refresh_slurm_activity

    printf 'ChannelSelectionPipeline monitor - %s\n' "$(date '+%Y-%m-%d %H:%M:%S %Z')"
    echo "Results: ${results_root}"
    echo "Scope: ${#datasets[@]} datasets x ${#transforms[@]} transforms x ${#classifiers[@]} classifiers"
    echo

    print_relevant_queue
    print_current_activity

    printf '%-27s %9s %4s %5s %7s %5s %4s %7s %7s %7s %10s\n' \
        "PIPELINE" "COMPLETE" "RUN" "QUEUE" "PENDING" "EMPTY" "OOM" \
        "FAILED" "LOGGED" "WAITING" "NOTSTARTED"
    printf '%-27s %9s %4s %5s %7s %5s %4s %7s %7s %7s %10s\n' \
        "---------------------------" "---------" "----" "-----" "-------" \
        "-----" "----" "-------" "-------" "-------" "----------"

    for classifier in "${classifiers[@]}"; do
        for transform in "${transforms[@]}"; do
            pipeline="${transform}-${classifier}"
            complete=0
            running=0
            queued=0
            pending=0
            empty=0
            oom=0
            failed=0
            logged=0
            waiting=0
            not_started=0

            for dataset in "${datasets[@]}"; do
                status_for "${pipeline}" "${dataset}"
                status="${status_result}"
                case "${status}" in
                    COMPLETE) complete=$((complete + 1)) ;;
                    RUNNING) running=$((running + 1)) ;;
                    QUEUED) queued=$((queued + 1)) ;;
                    PENDING) pending=$((pending + 1)) ;;
                    EMPTY) empty=$((empty + 1)) ;;
                    OOM) oom=$((oom + 1)) ;;
                    FAILED) failed=$((failed + 1)) ;;
                    LOGGED) logged=$((logged + 1)) ;;
                    WAITING) waiting=$((waiting + 1)) ;;
                    NOTSTARTED) not_started=$((not_started + 1)) ;;
                esac

                if [[ "${details}" == "true" && "${status}" != "COMPLETE" ]]; then
                    detail_key="${pipeline}|${dataset}"
                    detail_log="${slurm_combo_log[${detail_key}]-}"
                    if [[ -z "${detail_log}" ]]; then
                        latest_log_for "${pipeline}" "${dataset}"
                        detail_log="${latest_log_result}"
                    fi
                    if [[ -z "${detail_log}" ]]; then
                        detail_log="-"
                    fi
                    incomplete_details+=(
                        "${pipeline}|${dataset}|${status}|${detail_log}"
                    )
                fi
            done

            total_complete=$((total_complete + complete))
            total_running=$((total_running + running))
            total_queued=$((total_queued + queued))
            total_pending=$((total_pending + pending))
            total_empty=$((total_empty + empty))
            total_oom=$((total_oom + oom))
            total_failed=$((total_failed + failed))
            total_logged=$((total_logged + logged))
            total_waiting=$((total_waiting + waiting))
            total_not_started=$((total_not_started + not_started))

            printf '%-27s %3d/%-5d %4d %5d %7d %5d %4d %7d %7d %7d %10d\n' \
                "${pipeline}" "${complete}" "${expected_per_pipeline}" \
                "${running}" "${queued}" "${pending}" "${empty}" "${oom}" \
                "${failed}" "${logged}" "${waiting}" "${not_started}"
        done
    done

    echo
    printf 'Overall complete: %d/%d (%.1f%%)\n' \
        "${total_complete}" \
        "${total_expected}" \
        "$(awk -v done="${total_complete}" -v all="${total_expected}" \
            'BEGIN { printf 100 * done / all }')"
    echo "Incomplete status totals:"
    echo "  Currently running:       ${total_running}"
    echo "  Queued inside task farm: ${total_queued}"
    echo "  Pending Slurm jobs:      ${total_pending}"
    echo "  Empty result files:      ${total_empty}"
    echo "  Logged memory failures:  ${total_oom}"
    echo "  Other logged failures:   ${total_failed}"
    echo "  Output exists/no result: ${total_logged}"
    echo "  Empty output log:        ${total_waiting}"
    echo "  No output log:           ${total_not_started}"

    if [[ "${details}" == "true" ]]; then
        echo
        echo "Incomplete classifier/problem pairs"
        echo "-----------------------------------"
        printf '%-27s %-28s %-12s %s\n' \
            "PIPELINE" "DATASET" "STATUS" "EVIDENCE LOG"
        for line in "${incomplete_details[@]}"; do
            IFS='|' read -r pipeline dataset status detail_log <<< "${line}"
            printf '%-27s %-28s %-12s %s\n' \
                "${pipeline}" "${dataset}" "${status}" "${detail_log}"
        done
    else
        echo
        echo "Use --details to list individual incomplete datasets."
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
