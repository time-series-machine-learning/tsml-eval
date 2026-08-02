#!/bin/bash

set -uo pipefail

# Monitor the four GEAR-Comp train/test result sets used to construct
# component-aware V6 over the 25-problem EEG archive.

username="${USER:-ajb2u23}"
local_path="/iridisfs/home/${username}"
results_dir="${CHANNEL_SELECTION_RESULTS_ROOT:-${local_path}/Results/ChannelSelectionPipeline}"
resample=0

details="false"
watch_seconds=0

fast_datasets=(
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
)

slow_datasets=(
    "SitStand"
    "ShortIntervalTask"
    "MatchingPennies"
    "LongIntervalTask"
)

components=("Arsenal" "DrCIF" "STC" "TDE")

usage() {
    printf '%s\n' \
        "Usage:" \
        "  monitor_gear_component_train_files.sh [--details] [--watch SECONDS]" \
        "" \
        "Options:" \
        "  --details        List every incomplete component/dataset pair." \
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
                echo "ERROR: --watch requires a refresh interval." >&2
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

declare -A job_state=()
declare -A job_id=()
declare -A job_time=()
declare -A job_location=()
declare -a active_jobs=()
latest_log=""

refresh_jobs() {
    local id state elapsed name location key previous

    job_state=()
    job_id=()
    job_time=()
    job_location=()
    active_jobs=()

    while IFS='|' read -r id state elapsed name location; do
        [[ -z "${id}" ]] && continue
        [[ "${name}" != eeg-gear-comp-train-* ]] && continue

        active_jobs+=("${id}|${state}|${elapsed}|${name}|${location}")
        key="${name}"
        previous="${job_state[${key}]-}"

        # Prefer a running instance if an older duplicate is also pending.
        if [[ -z "${previous}" || "${state}" == "R" ]]; then
            job_state["${key}"]="${state}"
            job_id["${key}"]="${id}"
            job_time["${key}"]="${elapsed}"
            job_location["${key}"]="${location}"
        fi
    done < <(
        squeue --user="${username}" --noheader \
            --format="%i|%t|%M|%j|%R" 2>/dev/null || true
    )
}

find_latest_log() {
    local pipeline="$1"
    local component="$2"
    local group="$3"
    local dataset="$4"
    local output_dir="${results_dir}/output/${pipeline}"
    local candidate
    local -a candidates=()

    latest_log=""
    [[ ! -d "${output_dir}" ]] && return

    shopt -s nullglob
    candidates=(
        "${output_dir}"/output-"${dataset}"-"${resample}"-*-GEAR-Comp-"${component}"-"${group}"-"${group}".txt
    )
    shopt -u nullglob

    for candidate in "${candidates[@]}"; do
        if [[ -z "${latest_log}" || "${candidate}" -nt "${latest_log}" ]]; then
            latest_log="${candidate}"
        fi
    done
}

log_has_failure() {
    local log_file="$1"

    [[ -n "${log_file}" && -s "${log_file}" ]] || return 1
    grep -Eiq \
        'Traceback|MemoryError|OUT_OF_MEMORY|oom-kill|Killed|^ERROR:' \
        "${log_file}"
}

status_for() {
    local component="$1"
    local group="$2"
    local dataset="$3"
    local pipeline="GEAR-Comp-${component}"
    local prediction_dir="${results_dir}/${pipeline}/Predictions/${dataset}"
    local test_file="${prediction_dir}/testResample${resample}.csv"
    local train_file="${prediction_dir}/trainResample${resample}.csv"
    local job_name="eeg-gear-comp-train-${component,,}-${group}"
    local state="${job_state[${job_name}]-}"

    if [[ -s "${test_file}" && -s "${train_file}" ]]; then
        printf 'COMPLETE'
        return
    fi

    find_latest_log "${pipeline}" "${component}" "${group}" "${dataset}"
    if log_has_failure "${latest_log}"; then
        printf 'FAILED'
    elif [[ "${state}" == "PD" ]]; then
        printf 'PENDING'
    elif [[ "${state}" == "R" && -n "${latest_log}" ]]; then
        printf 'RUNNING'
    elif [[ "${state}" == "R" ]]; then
        printf 'WAITING'
    elif [[ -s "${test_file}" ]]; then
        printf 'TEST_ONLY'
    elif [[ -s "${train_file}" ]]; then
        printf 'TRAIN_ONLY'
    elif [[ -n "${latest_log}" ]]; then
        printf 'STARTED'
    else
        printf 'MISSING'
    fi
}

print_snapshot() {
    local component group dataset status
    local -a datasets=()
    local complete running waiting pending failed test_only train_only started missing
    local total_complete=0
    local expected=100
    local v6_ready=0

    refresh_jobs

    printf "GEAR-Comp train/test progress - %s\n" "$(date)"
    printf "Results: %s\n\n" "${results_dir}"

    printf "Active GEAR-Comp train jobs - %d\n" "${#active_jobs[@]}"
    printf '%-12s %-2s %-10s %-42s %s\n' \
        "JOBID" "ST" "TIME" "NAME" "NODE/REASON"
    if ((${#active_jobs[@]} == 0)); then
        printf '%s\n' "(none)"
    else
        for record in "${active_jobs[@]}"; do
            IFS='|' read -r id state elapsed name location <<< "${record}"
            printf '%-12s %-2s %-10s %-42s %s\n' \
                "${id}" "${state}" "${elapsed}" "${name}" "${location}"
        done
    fi

    printf '\n%-9s %-5s %9s %4s %4s %4s %6s %9s %10s %7s %7s\n' \
        "COMPONENT" "GROUP" "COMPLETE" "RUN" "WAIT" "PEND" "FAILED" \
        "TEST_ONLY" "TRAIN_ONLY" "STARTED" "MISSING"

    for component in "${components[@]}"; do
        for group in fast slow; do
            if [[ "${group}" == "fast" ]]; then
                datasets=("${fast_datasets[@]}")
            else
                datasets=("${slow_datasets[@]}")
            fi

            complete=0
            running=0
            waiting=0
            pending=0
            failed=0
            test_only=0
            train_only=0
            started=0
            missing=0

            for dataset in "${datasets[@]}"; do
                status=$(status_for "${component}" "${group}" "${dataset}")
                case "${status}" in
                    COMPLETE) complete=$((complete + 1)) ;;
                    RUNNING) running=$((running + 1)) ;;
                    WAITING) waiting=$((waiting + 1)) ;;
                    PENDING) pending=$((pending + 1)) ;;
                    FAILED) failed=$((failed + 1)) ;;
                    TEST_ONLY) test_only=$((test_only + 1)) ;;
                    TRAIN_ONLY) train_only=$((train_only + 1)) ;;
                    STARTED) started=$((started + 1)) ;;
                    MISSING) missing=$((missing + 1)) ;;
                esac
            done

            total_complete=$((total_complete + complete))
            printf '%-9s %-5s %4d/%-4d %4d %4d %4d %6d %9d %10d %7d %7d\n' \
                "${component}" "${group}" "${complete}" "${#datasets[@]}" \
                "${running}" "${waiting}" "${pending}" "${failed}" \
                "${test_only}" "${train_only}" "${started}" "${missing}"
        done
    done

    for dataset in "${fast_datasets[@]}" "${slow_datasets[@]}"; do
        ready=true
        for component in "${components[@]}"; do
            prediction_dir="${results_dir}/GEAR-Comp-${component}/Predictions/${dataset}"
            if [[ ! -s "${prediction_dir}/testResample${resample}.csv" ||
                  ! -s "${prediction_dir}/trainResample${resample}.csv" ]]; then
                ready=false
                break
            fi
        done
        [[ "${ready}" == true ]] && v6_ready=$((v6_ready + 1))
    done

    printf '\nOverall complete component pairs: %d/%d\n' \
        "${total_complete}" "${expected}"
    printf 'V6-from-file ready datasets:     %d/25\n' "${v6_ready}"

    if [[ "${details}" == "true" ]]; then
        printf '\nIncomplete pairs\n'
        printf '%-11s %-22s %-5s %s\n' "STATE" "PIPELINE" "GROUP" "DATASET"
        for component in "${components[@]}"; do
            for group in fast slow; do
                if [[ "${group}" == "fast" ]]; then
                    datasets=("${fast_datasets[@]}")
                else
                    datasets=("${slow_datasets[@]}")
                fi
                for dataset in "${datasets[@]}"; do
                    status=$(status_for "${component}" "${group}" "${dataset}")
                    if [[ "${status}" != "COMPLETE" ]]; then
                        printf '%-11s %-22s %-5s %s\n' \
                            "${status}" "GEAR-Comp-${component}" "${group}" "${dataset}"
                    fi
                done
            done
        done
    fi
}

while true; do
    if ((watch_seconds > 0)); then
        clear
    fi
    print_snapshot
    if ((watch_seconds == 0)); then
        break
    fi
    sleep "${watch_seconds}"
done
