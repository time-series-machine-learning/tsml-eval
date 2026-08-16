#!/bin/bash

set -euo pipefail

# Run the GEAR-Auto LOSO components with their native train estimates.
#
# One command per held-out subject fits the GEAR-Auto reduction once and then
# fits every component on it, so the channel selector is charged once per fold
# rather than once per component. Results are written below
# GEAR-Auto-Native-{Arsenal,DrCIF,STC} and the shared reduction cost to
# GEAR-Auto-Native-Reduction. Existing GEAR-Auto-HC2 and GEAR-Comp-* results are
# untouched.
#
# The default component set excludes TDE. Set COMPONENTS to override, e.g.
#   COMPONENTS="Arsenal DrCIF STC TDE" bash run_openclosefist_loso_gear_auto_native.sh
#
# To spread 105 folds over several nodes, submit ranges concurrently:
#   for r in "0 26" "27 53" "54 80" "81 104"; do
#       set -- $r
#       FIRST_SUBJECT=$1 LAST_SUBJECT=$2 bash run_openclosefist_loso_gear_auto_native.sh
#   done

first_subject="${FIRST_SUBJECT:-0}"
last_subject="${LAST_SUBJECT:-104}"
read -r -a components <<< "${COMPONENTS:-Arsenal DrCIF STC}"

username="ajb2u23"
local_path="/iridisfs/home/${username}"
queue="batch"
max_num_submitted=200
max_time="60:00:00"
mail="NONE"
mailto="${username}@soton.ac.uk"

dataset="OpenCloseFist"
result_dataset="${dataset}LOSO"

# Each task runs all components for one fold sequentially, so it must be sized
# for the most demanding of them. DrCIF needed 20 GiB per task in the GEAR-Comp
# runs; TDE needs 30 GiB if it is added back.
memory_per_cpu_gib=20
max_cpus_to_use=30
for component in "${components[@]}"; do
    if [[ "${component}" == "TDE" ]]; then
        memory_per_cpu_gib=30
        max_cpus_to_use=20
    fi
done

if ((first_subject < 0 || last_subject < first_subject || last_subject > 104)); then
    echo "ERROR: subject range must lie within 0..104." >&2
    exit 2
fi
if ((max_cpus_to_use * memory_per_cpu_gib > 620)); then
    echo "ERROR: requested memory exceeds the 620-GiB node safety ceiling." >&2
    exit 2
fi

tsml_eval_dir="${local_path}/Code/tsml-eval"
aeon_dir="${local_path}/Code/aeon"
python_path="/home/${username}/.conda/envs/tsml-eval/bin/python"
worker="${tsml_eval_dir}/tsml_eval/_wip/eeg_cote/run_gear_auto_loso_components.py"

data_root="${local_path}/Data/EEG"
results_root="${local_path}/Results/ChannelSelectionLOSO"
output_root="${results_root}/output"
numba_cache_dir="${local_path}/Code/.cache/tsml-eval"

for required in "${python_path}" "${worker}"; do
    if [[ ! -e "${required}" ]]; then
        echo "ERROR: required file is missing: ${required}" >&2
        exit 1
    fi
done
for suffix in TRAIN TEST; do
    if [[ ! -s "${data_root}/${dataset}/${dataset}_${suffix}.ts" \
          || ! -s "${data_root}/${dataset}/${dataset}_id_${suffix}.txt" ]]; then
        echo "ERROR: missing ${dataset} ${suffix} data or subject IDs." >&2
        exit 1
    fi
done
for repository in "${tsml_eval_dir}" "${aeon_dir}"; do
    if [[ ! -d "${repository}/.git" ]]; then
        echo "ERROR: Git checkout is missing: ${repository}" >&2
        exit 1
    fi
done

mkdir -p "${results_root}" "${output_root}" "${numba_cache_dir}"
tsml_eval_commit=$(git -C "${tsml_eval_dir}" rev-parse HEAD)
aeon_commit=$(git -C "${aeon_dir}" rev-parse HEAD)

# Fail before submission if the reducer or a component cannot be constructed.
PYTHONNOUSERSITE=1 PYTHONPATH="${aeon_dir}:${tsml_eval_dir}" \
"${python_path}" - "${components[@]}" <<'PY'
import sys
import aeon
import aeon_neuro
import tsml_eval
from tsml_eval.experiments._channel_selection_hc2 import _make_gear_transformer
from tsml_eval.experiments._get_classifier import _make_hc2_or_component

reducer = _make_gear_transformer(component="auto", random_state=0, n_jobs=1)
print("reducer:   ", type(reducer).__name__)
for component in sys.argv[1:]:
    estimator = _make_hc2_or_component(
        component=component.casefold(), random_state=0, n_jobs=1,
        fit_contract=0, kwargs={},
    )
    assert estimator.get_tag("capability:train_estimate")
    print(component, type(estimator).__name__)
print("aeon:      ", aeon.__file__)
print("aeon-neuro:", aeon_neuro.__file__)
print("tsml-eval: ", tsml_eval.__file__)
PY

run_id=$(date +%Y%m%d%H%M%S)
range_label="subjects-${first_subject}-${last_subject}"
submission_dir="${results_root}/batch-submissions/${run_id}-gear-auto-native-${range_label}"
command_file="${submission_dir}/generatedCommandList-${run_id}.txt"
submission_file="${submission_dir}/generatedSubmissionFile-${run_id}.sub"
mkdir -p "${submission_dir}"
: > "${command_file}"

log_dir="${output_root}/GEAR-Auto-Native"
mkdir -p "${log_dir}"

command_count=0
for ((subject=first_subject; subject<=last_subject; subject++)); do
    complete=1
    for component in "${components[@]}"; do
        prediction_dir="${results_root}/GEAR-Auto-Native-${component}/Predictions/${result_dataset}"
        if [[ ! -s "${prediction_dir}/trainResample${subject}.csv" \
              || ! -s "${prediction_dir}/testResample${subject}.csv" ]]; then
            complete=0
            break
        fi
    done
    if ((complete == 1)); then
        continue
    fi

    command=(
        "${python_path}" -u "${worker}"
        "${data_root}" "${results_root}" "${subject}"
        --dataset "${dataset}"
        --components "${components[@]}"
    )
    printf -v command_line '%q ' "${command[@]}"
    printf '%s> %q 2>&1\n' \
        "${command_line}" \
        "${log_dir}/output-${result_dataset}-${subject}-${run_id}.txt" \
        >> "${command_file}"
    command_count=$((command_count + 1))
done

if ((command_count == 0)); then
    echo "All selected GEAR-Auto native LOSO results already exist; no job submitted."
    exit 0
fi

cpu_count=${command_count}
if ((cpu_count > max_cpus_to_use)); then
    cpu_count=${max_cpus_to_use}
fi

cat > "${submission_file}" <<EOF
#!/bin/bash
#SBATCH --mail-type=${mail}
#SBATCH --mail-user=${mailto}
#SBATCH --job-name=eeg-ocf-gear-auto-native-${first_subject}-${last_subject}
#SBATCH --partition=${queue}
#SBATCH --time=${max_time}
#SBATCH --output=${submission_dir}/%A-${run_id}.out
#SBATCH --error=${submission_dir}/%A-${run_id}.err
#SBATCH --nodes=1
#SBATCH --ntasks=${cpu_count}
#SBATCH --mem-per-cpu=${memory_per_cpu_gib}G

. /etc/profile
set -eo pipefail
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
if [[ "\${current_tsml_eval_commit}" != "${tsml_eval_commit}" ]]; then
    echo "ERROR: tsml-eval changed after submission." >&2
    exit 1
fi
if [[ "\${current_aeon_commit}" != "${aeon_commit}" ]]; then
    echo "ERROR: aeon changed after submission." >&2
    exit 1
fi

echo "Components:       ${components[*]}"
echo "Subjects:         ${first_subject}..${last_subject}"
echo "Shared reduction: one GEAR-Auto fit per fold"
echo "Host:             \$(hostname)"
echo "Slurm job ID:     \${SLURM_JOB_ID}"
echo "Allocated tasks:  \${SLURM_NTASKS}"
echo "Command count:    ${command_count}"
echo "Memory per task:  ${memory_per_cpu_gib} GiB"
echo "Command file:     ${command_file}"
staskfarm "${command_file}"
EOF

while true; do
    num_jobs=$(squeue --noheader --user="${username}" --partition="${queue}" \
        --states=RUNNING,PENDING | wc -l)
    if ((num_jobs < max_num_submitted)); then
        break
    fi
    echo "Waiting 60 seconds: ${num_jobs} jobs are running or pending."
    sleep 60
done

sbatch_output=$(sbatch "${submission_file}")
echo "${sbatch_output}"
echo "Submitted ${command_count} GEAR-Auto native LOSO folds using ${cpu_count} tasks."
echo "Components ${components[*]} share one reduction per fold."
rm -f "${submission_file}"
