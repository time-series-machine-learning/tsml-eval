#!/bin/bash

set -euo pipefail

# Run the native-estimate GEAR-Comp LOSO family. Existing GEAR-Comp-* files use
# external ten-fold CV and remain untouched. New files are written below
# GEAR-Comp-Native-{Arsenal,DrCIF,STC,TDE}.

run_set="${RUN_SET:-fast}"
run_set="${run_set,,}"
first_subject="${FIRST_SUBJECT:-0}"
last_subject="${LAST_SUBJECT:-104}"

username="ajb2u23"
local_path="/iridisfs/home/${username}"
queue="batch"
max_num_submitted=200
max_time="60:00:00"
mail="NONE"
mailto="${username}@soton.ac.uk"

dataset="OpenCloseFist"
result_dataset="${dataset}LOSO"

case "${run_set}" in
    fast)
        components=("Arsenal" "STC")
        max_cpus_to_use=76
        memory_per_cpu_gib=8
        ;;
    arsenal)
        components=("Arsenal")
        max_cpus_to_use=76
        memory_per_cpu_gib=8
        ;;
    stc)
        components=("STC")
        max_cpus_to_use=76
        memory_per_cpu_gib=8
        ;;
    drcif)
        components=("DrCIF")
        max_cpus_to_use=30
        memory_per_cpu_gib=20
        ;;
    tde)
        components=("TDE")
        max_cpus_to_use=20
        memory_per_cpu_gib=30
        ;;
    *)
        echo "ERROR: RUN_SET must be fast, arsenal, stc, drcif, or tde." >&2
        exit 2
        ;;
esac

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
worker="${tsml_eval_dir}/tsml_eval/_wip/eeg_cote/run_native_gear_loso_component.py"

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

PYTHONNOUSERSITE=1 PYTHONPATH="${aeon_dir}:${tsml_eval_dir}" \
"${python_path}" - "${components[@]}" <<'PY'
import sys
import aeon
import aeon_neuro
import tsml_eval
from tsml_eval._wip.eeg_cote.run_native_gear_component import (
    GEARNativeComponentPipeline,
)

for component in sys.argv[1:]:
    estimator = GEARNativeComponentPipeline(component, random_state=0, n_jobs=1)
    assert estimator.get_tag("capability:train_estimate")
    print(component, type(estimator).__name__)
print("aeon:      ", aeon.__file__)
print("aeon-neuro:", aeon_neuro.__file__)
print("tsml-eval: ", tsml_eval.__file__)
PY

run_id=$(date +%Y%m%d%H%M%S)
range_label="subjects-${first_subject}-${last_subject}"
submission_dir="${results_root}/batch-submissions/${run_id}-gear-native-${run_set}-${range_label}"
command_file="${submission_dir}/generatedCommandList-${run_id}.txt"
submission_file="${submission_dir}/generatedSubmissionFile-${run_id}.sub"
mkdir -p "${submission_dir}"
: > "${command_file}"

command_count=0
for ((subject=first_subject; subject<=last_subject; subject++)); do
    for component in "${components[@]}"; do
        result_name="GEAR-Comp-Native-${component}"
        prediction_dir="${results_root}/${result_name}/Predictions/${result_dataset}"
        train_file="${prediction_dir}/trainResample${subject}.csv"
        test_file="${prediction_dir}/testResample${subject}.csv"
        if [[ -s "${train_file}" && -s "${test_file}" ]]; then
            continue
        fi

        log_dir="${output_root}/${result_name}"
        mkdir -p "${log_dir}"
        command=(
            "${python_path}" -u "${worker}"
            "${data_root}" "${results_root}" "${component}" "${subject}"
            --dataset "${dataset}"
        )
        printf -v command_line '%q ' "${command[@]}"
        printf '%s> %q 2>&1\n' \
            "${command_line}" \
            "${log_dir}/output-${result_dataset}-${subject}-${run_id}.txt" \
            >> "${command_file}"
        command_count=$((command_count + 1))
    done
done

if ((command_count == 0)); then
    echo "All selected native LOSO results already exist; no job submitted."
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
#SBATCH --job-name=eeg-ocf-gear-native-${run_set}-${first_subject}-${last_subject}
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

echo "Run set:          ${run_set}"
echo "Subjects:         ${first_subject}..${last_subject}"
echo "Native mechanism: enabled"
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
echo "Submitted ${command_count} native LOSO cells using ${cpu_count} tasks."
echo "Results remain separate below GEAR-Comp-Native-* directories."
rm -f "${submission_file}"
