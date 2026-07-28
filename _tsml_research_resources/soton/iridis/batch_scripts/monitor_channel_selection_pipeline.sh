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
results_root="${local_path}/Results/ChannelSelectionPipeline"
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

# Include every transform family used in the original pipeline experiments and
# the subsequent guarded-reducer revisions. Remove a row here only when that
# family has deliberately been retired from the result archive.
transforms=(
    "CSP"
    "ECS"
    "ECP"
    "TSelect"
    "Random"
    "Riemannian"
    "DetachRocket"
    "CaseTimeReducer"
    "GuardedMultiAxis"
    "GMARv2"
    "CLeVerRank"
    "CLeVerCluster"
    "CLeVerHybrid"
    "GMARv3"
    "GMARv4"
)

# HC2 and the four classifiers from which it is built.
classifiers=(
    "HC2"
    "Arsenal"
    "DrCIF"
    "STC"
    "TDE"
)

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

    printf '%s' "${latest}"
}

status_for() {
    local pipeline="$1"
    local dataset="$2"
    local result_file
    local latest_log

    result_file="${results_root}/${pipeline}/Predictions/${dataset}/testResample${resample}.csv"

    if [[ -s "${result_file}" ]]; then
        printf 'COMPLETE'
        return
    fi
    if [[ -e "${result_file}" ]]; then
        printf 'EMPTY'
        return
    fi

    latest_log=$(latest_log_for "${pipeline}" "${dataset}")
    if [[ -z "${latest_log}" ]]; then
        printf 'NOTSTARTED'
        return
    fi
    if [[ ! -s "${latest_log}" ]]; then
        printf 'WAITING'
        return
    fi

    if grep -Eiq \
        'out[ -]?of[ -]?memory|OUT_OF_MEMORY|oom[_-]kill|Killed process|MemoryError|Cannot allocate memory|std::bad_alloc' \
        "${latest_log}"; then
        printf 'OOM'
        return
    fi
    if grep -Eiq \
        'Traceback \(most recent call last\)|Segmentation fault|^ERROR:|slurmstepd: error:|Exception:' \
        "${latest_log}"; then
        printf 'FAILED'
        return
    fi

    # A non-empty output log exists, but a result has not appeared and the log
    # has no recognised failure. It may be running, queued for a retry, or have
    # stopped without a conventional Python/Slurm error.
    printf 'LOGGED'
}

print_relevant_queue() {
    local queue_output
    local line
    local lower_line
    local found=0

    echo "Relevant Slurm jobs"
    echo "-------------------"

    if ! command -v squeue >/dev/null 2>&1; then
        echo "squeue is unavailable in this shell."
        echo
        return
    fi

    queue_output=$(
        squeue \
            --noheader \
            --user="${username}" \
            --format="%.18i %.9P %.32j %.2t %.10M %.6D %R" \
            2>/dev/null
    )

    while IFS= read -r line; do
        [[ -z "${line}" ]] && continue
        lower_line="${line,,}"
        if [[ "${lower_line}" == *eeg* \
            || "${lower_line}" == *channel* \
            || "${lower_line}" == *gmar* ]]; then
            if ((found == 0)); then
                printf '%-18s %-9s %-32s %-2s %-10s %-6s %s\n' \
                    "JOBID" "PARTITION" "NAME" "ST" "TIME" "NODES" "NODELIST(REASON)"
            fi
            echo "${line}"
            found=1
        fi
    done <<< "${queue_output}"

    if ((found == 0)); then
        echo "No matching running or pending jobs."
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
    local total_empty=0
    local total_oom=0
    local total_failed=0
    local total_logged=0
    local total_waiting=0
    local total_not_started=0
    local -a incomplete_details=()

    if [[ ! -d "${results_root}" ]]; then
        echo "ERROR: results directory not found:"
        echo "  ${results_root}"
        return 1
    fi

    printf 'ChannelSelectionPipeline monitor — %s\n' "$(date '+%Y-%m-%d %H:%M:%S %Z')"
    echo "Results: ${results_root}"
    echo "Scope: ${#datasets[@]} datasets x ${#transforms[@]} transforms x ${#classifiers[@]} classifiers"
    echo

    print_relevant_queue

    printf '%-27s %9s %5s %5s %7s %7s %7s %10s\n' \
        "PIPELINE" "COMPLETE" "EMPTY" "OOM" "FAILED" "LOGGED" "WAITING" "NOTSTARTED"
    printf '%-27s %9s %5s %5s %7s %7s %7s %10s\n' \
        "---------------------------" "---------" "-----" "-----" "-------" "-------" "-------" "----------"

    for classifier in "${classifiers[@]}"; do
        for transform in "${transforms[@]}"; do
            pipeline="${transform}-${classifier}"
            complete=0
            empty=0
            oom=0
            failed=0
            logged=0
            waiting=0
            not_started=0

            for dataset in "${datasets[@]}"; do
                status=$(status_for "${pipeline}" "${dataset}")
                case "${status}" in
                    COMPLETE) complete=$((complete + 1)) ;;
                    EMPTY) empty=$((empty + 1)) ;;
                    OOM) oom=$((oom + 1)) ;;
                    FAILED) failed=$((failed + 1)) ;;
                    LOGGED) logged=$((logged + 1)) ;;
                    WAITING) waiting=$((waiting + 1)) ;;
                    NOTSTARTED) not_started=$((not_started + 1)) ;;
                esac

                if [[ "${details}" == "true" && "${status}" != "COMPLETE" ]]; then
                    incomplete_details+=("${pipeline}|${dataset}|${status}")
                fi
            done

            total_complete=$((total_complete + complete))
            total_empty=$((total_empty + empty))
            total_oom=$((total_oom + oom))
            total_failed=$((total_failed + failed))
            total_logged=$((total_logged + logged))
            total_waiting=$((total_waiting + waiting))
            total_not_started=$((total_not_started + not_started))

            printf '%-27s %3d/%-5d %5d %5d %7d %7d %7d %10d\n' \
                "${pipeline}" "${complete}" "${expected_per_pipeline}" \
                "${empty}" "${oom}" "${failed}" "${logged}" "${waiting}" \
                "${not_started}"
        done
    done

    echo
    printf 'Overall complete: %d/%d (%.1f%%)\n' \
        "${total_complete}" \
        "${total_expected}" \
        "$(awk -v done="${total_complete}" -v all="${total_expected}" \
            'BEGIN { printf 100 * done / all }')"
    echo "Incomplete status totals:"
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
        printf '%-27s %-28s %s\n' "PIPELINE" "DATASET" "STATUS"
        for line in "${incomplete_details[@]}"; do
            IFS='|' read -r pipeline dataset status <<< "${line}"
            printf '%-27s %-28s %s\n' "${pipeline}" "${dataset}" "${status}"
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
