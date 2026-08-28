#!/usr/bin/env bash
# Run the four outstanding Multiverse Core TDE resample-0 experiments without Slurm.
#
# The default is deliberately one experiment at a time because these datasets can use
# substantial memory. Set TDE_PARALLEL_JOBS to a larger value before `start` only when
# the machine has enough RAM for several fits at once.

set -euo pipefail

mode="${1:-start}"
script_path="$(readlink -f "${BASH_SOURCE[0]}")"
script_dir="$(cd "$(dirname "${script_path}")" && pwd)"
repo_dir="${TSML_EVAL_REPO_DIR:-$(cd "${script_dir}/.." && pwd)}"
data_dir="${MULTIVERSE_DATA_DIR:-${HOME}/Data/Multiverse}"
results_root="${MULTIVERSE_RESULTS_ROOT:-${HOME}/Results/Multiverse}"
results_dir="${results_root}/DictionaryBased"
state_dir="${TDE_UNIX_STATE_DIR:-${results_root}/.tde-core-missing-unix}"
log_dir="${state_dir}/logs"
pid_file="${state_dir}/runner.pid"
main_log="${state_dir}/runner.log"
lock_dir="${state_dir}/runner.lock"
parallel_jobs="${TDE_PARALLEL_JOBS:-1}"
expected_branch="${TSML_EVAL_EXPECTED_BRANCH:-ajb/hc2}"

datasets=(
    STEW
    USCActivity
    Tiselac
    AustraliaRainfall_disc
)

die() {
    echo "ERROR: $*" >&2
    exit 1
}

pid_is_running() {
    local pid="${1:-}"
    [[ "${pid}" =~ ^[0-9]+$ ]] && kill -0 "${pid}" 2>/dev/null
}

result_file() {
    local split="$1"
    local dataset="$2"
    printf '%s/TDE/Predictions/%s/%sResample0.csv' \
        "${results_dir}" "${dataset}" "${split}"
}

show_status() {
    local pid=""
    if [[ -f "${pid_file}" ]]; then
        pid="$(<"${pid_file}")"
    fi

    if pid_is_running "${pid}"; then
        echo "Runner: active (PID ${pid})"
    else
        echo "Runner: not active"
    fi

    echo "Results: ${results_dir}"
    echo "Log:     ${main_log}"
    echo

    local dataset test_state train_state run_state
    for dataset in "${datasets[@]}"; do
        test_state="missing"
        train_state="missing"
        run_state=""
        [[ -s "$(result_file test "${dataset}")" ]] && test_state="complete"
        [[ -s "$(result_file train "${dataset}")" ]] && train_state="complete"
        [[ -f "${state_dir}/running.${dataset}" ]] && run_state=" running"
        printf '%-30s test=%-8s train=%-8s%s\n' \
            "${dataset}" "${test_state}" "${train_state}" "${run_state}"
    done
}

validate_setup() {
    [[ "${parallel_jobs}" =~ ^[1-9][0-9]*$ ]] || \
        die "TDE_PARALLEL_JOBS must be a positive integer"
    [[ -d "${repo_dir}/tsml_eval" ]] || die "tsml-eval checkout not found: ${repo_dir}"
    [[ -d "${data_dir}" ]] || die "Multiverse data directory not found: ${data_dir}"

    local branch
    branch="$(git -C "${repo_dir}" branch --show-current)"
    if [[ -n "${expected_branch}" && "${branch}" != "${expected_branch}" ]]; then
        die "expected branch ${expected_branch}, but ${repo_dir} is on ${branch}"
    fi

    command -v python >/dev/null 2>&1 || die "python is not available; activate tsml-eval"
    command -v setsid >/dev/null 2>&1 || die "setsid is required (normally provided by util-linux)"
    (cd "${repo_dir}" && python -c "import aeon, tsml_eval") >/dev/null 2>&1 || \
        die "python cannot import aeon and tsml_eval; activate the tsml-eval environment"
}

run_dataset() {
    local dataset="$1"
    local test_file train_file dataset_log rc
    test_file="$(result_file test "${dataset}")"
    train_file="$(result_file train "${dataset}")"
    dataset_log="${log_dir}/TDE_${dataset}_resample0.log"

    if [[ -e "${test_file}" && ! -s "${test_file}" ]]; then
        echo "ERROR: ${test_file} exists but is empty; move or remove it before retrying." >&2
        return 2
    fi
    if [[ -e "${train_file}" && ! -s "${train_file}" ]]; then
        echo "ERROR: ${train_file} exists but is empty; move or remove it before retrying." >&2
        return 2
    fi

    if [[ -s "${test_file}" && -s "${train_file}" ]]; then
        echo "$(date --iso-8601=seconds) ${dataset}: test and train files exist; skipping."
        return 0
    fi

    touch "${state_dir}/running.${dataset}"
    {
        echo
        echo "============================================================================="
        echo "Started:    $(date --iso-8601=seconds)"
        echo "Host:       $(hostname)"
        echo "Repository: ${repo_dir}"
        echo "Revision:   $(git -C "${repo_dir}" rev-parse HEAD)"
        echo "Python:     $(command -v python)"
        echo "Dataset:    ${dataset}"
        echo "Results:    ${results_dir}"
        echo "============================================================================="
    } | tee -a "${dataset_log}"

    set +e
    (
        cd "${repo_dir}"
        export PYTHONUNBUFFERED=1
        export PYTHONWARNINGS="${PYTHONWARNINGS:-ignore}"
        export OMP_NUM_THREADS=1
        export MKL_NUM_THREADS=1
        export OPENBLAS_NUM_THREADS=1
        export NUMEXPR_NUM_THREADS=1
        export LOKY_MAX_CPU_COUNT=1
        python -u -m tsml_eval.experiments.classification_experiments \
            "${data_dir}" \
            "${results_dir}" \
            TDE \
            "${dataset}" \
            0 \
            -tr \
            -kw verbose 2 int
    ) 2>&1 | tee -a "${dataset_log}"
    rc="${PIPESTATUS[0]}"
    set -e

    rm -f "${state_dir}/running.${dataset}"
    echo "Finished: $(date --iso-8601=seconds); dataset=${dataset}; exit=${rc}" | \
        tee -a "${dataset_log}"
    return "${rc}"
}

run_all() {
    validate_setup
    mkdir -p "${state_dir}" "${log_dir}" "${results_dir}"

    if ! mkdir "${lock_dir}" 2>/dev/null; then
        die "another runner owns ${lock_dir}"
    fi

    echo "$$" > "${pid_file}"
    trap 'rmdir "${lock_dir}" 2>/dev/null || true; rm -f "${pid_file}"; rm -f "${state_dir}"/running.*' EXIT
    trap 'exit 130' INT TERM

    echo "TDE Unix runner started: $(date --iso-8601=seconds)"
    echo "Host:          $(hostname)"
    echo "Parallel jobs: ${parallel_jobs}"
    echo "Datasets:      ${datasets[*]}"

    local dataset pid
    local -a active_pids=()
    local failures=0

    for dataset in "${datasets[@]}"; do
        while (( ${#active_pids[@]} >= parallel_jobs )); do
            pid="${active_pids[0]}"
            if ! wait "${pid}"; then
                failures=$((failures + 1))
            fi
            active_pids=("${active_pids[@]:1}")
        done

        run_dataset "${dataset}" &
        active_pids+=("$!")
    done

    for pid in "${active_pids[@]}"; do
        if ! wait "${pid}"; then
            failures=$((failures + 1))
        fi
    done

    echo "TDE Unix runner finished: $(date --iso-8601=seconds); failures=${failures}"
    show_status
    (( failures == 0 ))
}

start_runner() {
    validate_setup
    mkdir -p "${state_dir}" "${log_dir}" "${results_dir}"

    local old_pid=""
    [[ -f "${pid_file}" ]] && old_pid="$(<"${pid_file}")"
    if pid_is_running "${old_pid}"; then
        die "runner is already active with PID ${old_pid}"
    fi

    rm -f "${pid_file}"
    rmdir "${lock_dir}" 2>/dev/null || true
    echo "Starting detached TDE runner on $(hostname)."
    echo "Output will be appended to ${main_log}"
    nohup setsid bash "${script_path}" run >> "${main_log}" 2>&1 < /dev/null &
    local launcher_pid=$!
    local pid=""
    local attempt
    for attempt in 1 2 3 4 5; do
        sleep 1
        [[ -f "${pid_file}" ]] && pid="$(<"${pid_file}")"
        pid_is_running "${pid}" && break
    done

    if ! pid_is_running "${pid}"; then
        wait "${launcher_pid}" 2>/dev/null || true
        tail -n 40 "${main_log}" 2>/dev/null || true
        die "runner exited during startup"
    fi

    echo "Started PID ${pid}."
    show_status
}

stop_runner() {
    local pid=""
    [[ -f "${pid_file}" ]] && pid="$(<"${pid_file}")"
    if ! pid_is_running "${pid}"; then
        echo "No active runner found."
        rm -f "${pid_file}"
        return 0
    fi

    echo "Stopping runner PID ${pid} and its active experiment."
    kill -TERM -- "-${pid}" 2>/dev/null || kill -TERM "${pid}"
}

case "${mode}" in
    start)
        start_runner
        ;;
    run)
        run_all
        ;;
    status)
        mkdir -p "${state_dir}"
        show_status
        ;;
    stop)
        stop_runner
        ;;
    *)
        die "usage: $(basename "${script_path}") {start|run|status|stop}"
        ;;
esac
