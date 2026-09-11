#!/bin/bash
set -euo pipefail
script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
config_file="${UCR_REFERENCE_CONFIG:-${script_dir}/ucr_reference.json}"
python_executable="${UCR_REFERENCE_PYTHON:-/home/ajb2u23/.conda/envs/tsml-eval/bin/python}"
once=false
stop=false
for argument in "$@"; do
    case "${argument}" in
        --once) once=true ;;
        --stop) stop=true ;;
        -h|--help) echo 'Usage: mail_ucr_reference_progress.sh [--once] [--stop]'; exit 0 ;;
        *) echo "ERROR: unknown argument: ${argument}" >&2; exit 2 ;;
    esac
done
readarray -t config_values < <("${python_executable}" - "${config_file}" <<'PY'
import json, sys
with open(sys.argv[1], encoding="utf-8") as stream:
    config = json.load(stream)
print(config["results_root"])
print(config.get("email", ""))
print(config.get("job_prefix", "ucr-reference"))
PY
)
results_root="${config_values[0]}"
email="${UCR_REFERENCE_EMAIL:-${config_values[1]}}"
job_prefix="${config_values[2]}"
state_dir="${results_root}/.ucr-reference-state"
reporter_dir="${state_dir}/reporter"
mkdir -p "${reporter_dir}"
stop_file="${reporter_dir}/STOP"
if [[ "${stop}" == true ]]; then : > "${stop_file}"; fi
if [[ -z "${email}" ]]; then echo "ERROR: configure email or UCR_REFERENCE_EMAIL" >&2; exit 1; fi
stamp=$(date '+%Y%m%d-%H%M%S')
report_file="${reporter_dir}/report-${stamp}.txt"
"${python_executable}" -u "${script_dir}/controller.py" monitor --config "${config_file}" > "${report_file}" 2>&1 || true
complete_line=$(grep -m1 '^Complete:' "${report_file}" || true)
subject="UCR reference progress [$(hostname -s)]: ${complete_line:-status unavailable}"
mailer=""
for candidate in mail mailx sendmail; do
    if command -v "${candidate}" >/dev/null 2>&1; then mailer="${candidate}"; break; fi
done
case "${mailer}" in
    mail|mailx) "${mailer}" -s "${subject}" "${email}" < "${report_file}" || true ;;
    sendmail) { printf 'To: %s\nSubject: %s\n\n' "${email}" "${subject}"; cat "${report_file}"; } | sendmail -t || true ;;
    *) echo "No mail command found; report saved at ${report_file}." ;;
esac
echo "Report: ${report_file}"
if [[ "${once}" == true || -f "${stop_file}" ]]; then exit 0; fi
if grep -qE '^Complete: [0-9,]+/[0-9,]+; percentage: 100\.00%; missing: 0;' "${report_file}"; then exit 0; fi
job_file="${reporter_dir}/daily-reporter.sub"
cat > "${job_file}" <<SUB
#!/bin/bash
#SBATCH --job-name=${job_prefix}-daily-report
#SBATCH --partition=batch
#SBATCH --time=00:10:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=2G
#SBATCH --mail-type=NONE
#SBATCH --output=${reporter_dir}/%j-daily.out
#SBATCH --error=${reporter_dir}/%j-daily.err
#SBATCH --begin=now+24hours
. /etc/profile
set -euo pipefail
bash "${script_dir}/mail_ucr_reference_progress.sh"
SUB
next=$(sbatch --parsable "${job_file}")
echo "Next daily report scheduled: ${next}"
