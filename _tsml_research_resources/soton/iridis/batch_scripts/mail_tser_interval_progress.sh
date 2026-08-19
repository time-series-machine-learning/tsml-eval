#!/bin/bash

set -euo pipefail

# Mail a progress report for the TSER interval regressor run every 12 hours.
#
# The run itself reports once per round, but a round can last the full 60 hour
# wall clock, so those messages say nothing for days at a time. This sends the
# monitor's output on a fixed clock instead.
#
# Start it once, after launching the run:
#
#   bash mail_tser_interval_progress.sh
#
# Send one report right now and schedule nothing:
#
#   bash mail_tser_interval_progress.sh --once
#
# Stop the cycle:
#
#   bash mail_tser_interval_progress.sh --stop
#
# Scheduling. Each report is a short Slurm job submitted with --begin, so
# nothing sits in an allocation waiting: the job is pending until its start
# time, runs for well under a minute, mails, and submits the next one. The
# cycle stops on its own when every experiment is complete.
#
# Node limits. The report job asks for one CPU, but if the four node limit
# counts jobs rather than cores it will occupy one of the four slots while it is
# pending. If that turns out to squeeze the run, use --mode local, which runs
# the same cycle as a background process on the login node and touches the
# scheduler not at all.

username="ajb2u23"
mailto="${mailto:-${username}@soton.ac.uk}"

local_path="/iridisfs/home/${username}"
results_root="${TSER_INTERVAL_RESULTS_ROOT:-${local_path}/Results/TSER/IntervalBased}"
state_dir="${results_root}/.tser-interval-state"
reporter_dir="${state_dir}/reporter"
stop_file="${reporter_dir}/stop"

queue="batch"
report_time="00:10:00"
report_memory="4G"

interval_hours="${interval_hours:-12}"
mode="slurm"
once="false"
stop="false"

script_path="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/$(basename "${BASH_SOURCE[0]}")"
monitor_path="$(dirname "${script_path}")/monitor_tser_interval_regressors.sh"

usage() {
    printf '%s\n' \
        "Usage:" \
        "  mail_tser_interval_progress.sh [options]" \
        "" \
        "Options:" \
        "  --interval-hours N  Hours between reports (default 12)." \
        "  --mode slurm|local  Schedule with Slurm, or loop on the login node." \
        "  --once              Send one report now and schedule nothing." \
        "  --stop              Stop the cycle after the next report." \
        "  -h, --help          Show this help."
}

while (($# > 0)); do
    case "$1" in
        --interval-hours)
            if (($# < 2)); then
                echo "ERROR: --interval-hours requires a value." >&2
                exit 2
            fi
            interval_hours="$2"
            shift 2
            ;;
        --mode)
            if (($# < 2)); then
                echo "ERROR: --mode requires slurm or local." >&2
                exit 2
            fi
            mode="$2"
            shift 2
            ;;
        --once)
            once="true"
            shift
            ;;
        --stop)
            stop="true"
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

if ! [[ "${interval_hours}" =~ ^[0-9]+$ ]] || ((interval_hours < 1)); then
    echo "ERROR: --interval-hours must be a positive integer." >&2
    exit 2
fi

if [[ "${mode}" != "slurm" && "${mode}" != "local" ]]; then
    echo "ERROR: --mode must be slurm or local." >&2
    exit 2
fi

if [[ ! -f "${monitor_path}" ]]; then
    echo "ERROR: monitor script not found:"
    echo "  ${monitor_path}"
    exit 1
fi

mkdir -p "${reporter_dir}"

if [[ "${stop}" == "true" ]]; then
    : > "${stop_file}"
    echo "Stop requested. The next report will send and then end the cycle."
    echo "Delete ${stop_file} to resume."
    exit 0
fi

send_mail() {
    local subject="$1"
    local body_file="$2"
    local mailer=""
    local candidate

    for candidate in mail mailx sendmail; do
        if command -v "${candidate}" >/dev/null 2>&1; then
            mailer="${candidate}"
            break
        fi
    done

    case "${mailer}" in
        mail|mailx)
            "${mailer}" -s "${subject}" "${mailto}" < "${body_file}"
            ;;
        sendmail)
            {
                printf 'To: %s\n' "${mailto}"
                printf 'Subject: %s\n\n' "${subject}"
                cat "${body_file}"
            } | sendmail -t
            ;;
        *)
            echo "No mail command found; report saved at ${body_file}."
            return 1
            ;;
    esac
}

# Returns 0 while experiments remain, 1 once everything is complete. The count
# comes from the monitor, which reads the result files rather than any state
# this script keeps.
send_report() {
    local stamp
    local report_file
    local overall
    local subject

    stamp=$(date '+%Y%m%d-%H%M%S')
    report_file="${reporter_dir}/report-${stamp}.txt"

    TSER_INTERVAL_RESULTS_ROOT="${results_root}" \
        bash "${monitor_path}" --summary > "${report_file}" 2>&1 || true

    overall=$(
        grep -m1 '^Overall complete:' "${report_file}" |
            sed 's/^Overall complete: //'
    )
    if [[ -z "${overall}" ]]; then
        overall="progress unknown"
    fi

    subject="TSER intervals: ${overall}"
    send_mail "${subject}" "${report_file}" || true

    # "13230/13230 (100.0%)" means there is nothing left to report on.
    if [[ "${overall}" == *"(100.0%)"* ]]; then
        return 1
    fi
    return 0
}

schedule_next() {
    local job_file="${reporter_dir}/reporter.sub"
    local sbatch_output

    if [[ -f "${stop_file}" ]]; then
        echo "Stop file present; not scheduling another report."
        return
    fi

    cat > "${job_file}" <<SUB
#!/bin/bash
#SBATCH --mail-type=NONE
#SBATCH --job-name=tser-interval-report
#SBATCH --partition=${queue}
#SBATCH --time=${report_time}
#SBATCH --output=${reporter_dir}/%A-report.out
#SBATCH --error=${reporter_dir}/%A-report.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=${report_memory}
#SBATCH --begin=now+${interval_hours}hours

. /etc/profile
set -e

bash "${script_path}" --interval-hours ${interval_hours} --mode slurm
SUB

    sbatch_output=$(sbatch "${job_file}")
    echo "Next report in ${interval_hours} hours: ${sbatch_output}"
}

if [[ "${mode}" == "local" ]]; then
    # A plain loop for when the scheduler should not be involved. Run it with
    # nohup so it survives logout:
    #   nohup bash mail_tser_interval_progress.sh --mode local &
    while true; do
        if ! send_report; then
            echo "Run complete; ending the report cycle."
            exit 0
        fi
        if [[ -f "${stop_file}" ]]; then
            echo "Stop file present; ending the report cycle."
            exit 0
        fi
        sleep $((interval_hours * 3600))
    done
fi

# Slurm mode. Send now, then schedule the next unless the run has finished.
if send_report; then
    if [[ "${once}" != "true" ]]; then
        schedule_next
    fi
else
    echo "Run complete; no further reports scheduled."
fi
