#!/bin/bash
# Restart the one-shot Multiverse controller every 30 minutes.
#
# Run with bash, not sh. Invoking it as "sh run_multiverse_controller.sh" runs bash in
# POSIX mode, which disables the process substitution used below and fails partway
# through startup with "syntax error near unexpected token `<'". The guard immediately
# below re-executes under bash so that mistake is corrected rather than fatal.
if [ -z "${BASH_VERSION-}" ]; then
    exec bash "$0" "$@"
fi
case ${SHELLOPTS-} in
    *posix*) exec bash "$0" "$@" ;;
esac

set -uo pipefail

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
repo_dir=$(cd -- "${script_dir}/.." && pwd)
config_file=${1:-"${script_dir}/multiverse_controller.toml"}
interval_seconds=${2:-${MULTIVERSE_CONTROLLER_INTERVAL_SECONDS:-1800}}
email_interval_seconds=${3:-${MULTIVERSE_EMAIL_INTERVAL_SECONDS:-14400}}
clear_pending_on_start=${MULTIVERSE_CLEAR_PENDING_ON_START:-true}
# The supervisor runs on a login node with no environment active, where "python" may
# not exist at all. Resolve one up front and stop if there is none, rather than
# looping every 30 minutes on "python: command not found".
#
# Note the interpreter needs Python 3.11 or newer for tomllib, and must be able to
# import tsml_eval when the configuration sets validate_results, so the experiment
# environment's interpreter is usually the right choice:
#
#   PYTHON=~/.conda/envs/tsml-eval-gpu/bin/python bash run_multiverse_controller.sh ...
python_executable=${PYTHON:-}
if [[ -z "${python_executable}" ]]; then
    for candidate in python python3; do
        if command -v "${candidate}" >/dev/null 2>&1; then
            python_executable=${candidate}
            break
        fi
    done
fi
if ! command -v "${python_executable:-}" >/dev/null 2>&1; then
    echo "ERROR: no Python interpreter found." >&2
    echo "Set PYTHON to one, or module load a Python before starting. e.g." >&2
    echo "  PYTHON=~/.conda/envs/tsml-eval-gpu/bin/python bash $0 <config>" >&2
    exit 1
fi
# Derived from $HOME so it is correct on any cluster. On Hali this resolves to the
# previous /gpfs/home/$USER path. Override with MULTIVERSE_LOG_DIR if the results live
# somewhere other than ~/Results/Multiverse
log_dir=${MULTIVERSE_LOG_DIR:-"${HOME}/Results/Multiverse/.controller"}
log_file="${log_dir}/supervisor.log"

if [[ ! "${interval_seconds}" =~ ^[1-9][0-9]*$ ]]; then
    echo "ERROR: interval must be a positive number of seconds" >&2
    exit 1
fi

if [[ ! "${email_interval_seconds}" =~ ^[1-9][0-9]*$ ]]; then
    echo "ERROR: email interval must be a positive number of seconds" >&2
    exit 1
fi

if [[ "${clear_pending_on_start}" != true &&
    "${clear_pending_on_start}" != false ]]; then
    echo "ERROR: MULTIVERSE_CLEAR_PENDING_ON_START must be true or false" >&2
    exit 1
fi

if [[ ! -f "${config_file}" ]]; then
    echo "ERROR: controller configuration not found: ${config_file}" >&2
    exit 1
fi

if ! mkdir -p "${log_dir}"; then
    echo "ERROR: could not create log directory: ${log_dir}" >&2
    echo "Set MULTIVERSE_LOG_DIR to a writable path." >&2
    exit 1
fi
cd "${repo_dir}" || exit 1

echo "Multiverse controller supervisor started."
echo "Configuration: ${config_file}"
echo "Cycle interval: ${interval_seconds} seconds"
echo "Email interval: ${email_interval_seconds} seconds"
echo "Clear pending jobs on start: ${clear_pending_on_start}"
echo "Log: ${log_file}"

if [[ "${clear_pending_on_start}" == true ]]; then
    if ! command -v squeue >/dev/null 2>&1 ||
        ! command -v scancel >/dev/null 2>&1; then
        echo "ERROR: squeue and scancel are required to clear pending jobs" >&2
        exit 1
    fi

    pending_output=$(
        squeue --noheader --array --user="${USER}" --states=PENDING \
            --format='%i'
    )
    mapfile -t pending_ids < <(
        sed -e 's/^[[:space:]]*//' -e 's/[[:space:]]*$//' -e '/^$/d' \
            <<< "${pending_output}"
    )
    if ((${#pending_ids[@]})); then
        echo "Cancelling ${#pending_ids[@]} pending Slurm tasks before first cycle." \
            | tee -a "${log_file}"
        scancel "${pending_ids[@]}"
    else
        echo "No pending Slurm tasks to cancel before first cycle." \
            | tee -a "${log_file}"
    fi
fi

# Send a true startup snapshot before the first queue-refill cycle. An interval
# of zero forces this report on every supervisor start; a successful send then
# records the normal four-hour email marker used by subsequent cycles.
echo "Sending initial controller state." | tee -a "${log_file}"
"${python_executable}" -u "${script_dir}/multiverse_controller.py" \
    --config "${config_file}" \
    --report-only \
    --email-interval-seconds 0 \
    2>&1 | tee -a "${log_file}"
initial_report_status=${PIPESTATUS[0]}
if ((initial_report_status != 0)); then
    echo "Initial controller report exited ${initial_report_status}; continuing." \
        | tee -a "${log_file}"
fi

while true; do
    echo "Controller cycle started: $(date --iso-8601=seconds)" | tee -a "${log_file}"
    "${python_executable}" -u "${script_dir}/multiverse_controller.py" \
        --config "${config_file}" \
        --email-interval-seconds "${email_interval_seconds}" \
        2>&1 | tee -a "${log_file}"
    controller_status=${PIPESTATUS[0]}
    echo "Controller exited ${controller_status}; restarting after ${interval_seconds}s." \
        | tee -a "${log_file}"
    sleep "${interval_seconds}"
done
