#!/bin/bash

set -euo pipefail

# Fit the two missing GEAR-Auto transforms independently of HC2. The resulting
# TRAIN/TEST collections are stored under ~/Results/Transforms and may be reused
# by run_missing_gear_transformed_classifiers.sh.

username="ajb2u23"
local_path="/iridisfs/home/${username}"
queue="batch"
max_num_submitted=200
max_time="60:00:00"
memory_per_cpu="30G"
mail="NONE"
mailto="${username}@soton.ac.uk"

transform_name="GEAR-Auto"
problems=(
    "FaceDetection"
    "LongIntervalTask"
)

tsml_eval_dir="${local_path}/Code/tsml-eval"
aeon_dir="${local_path}/Code/aeon"
python_path="/home/${username}/.conda/envs/tsml-eval/bin/python"
worker="${tsml_eval_dir}/tsml_eval/_wip/eeg_cote/generate_archive_transform.py"

data_root="${local_path}/Data/EEG"
transform_root="${local_path}/Results/Transforms"
output_root="${transform_root}/output/${transform_name}"
numba_cache_dir="${local_path}/Code/.cache/tsml-eval"

for required in "${python_path}" "${worker}"; do
    if [[ ! -e "${required}" ]]; then
        echo "ERROR: required file is missing: ${required}" >&2
        exit 1
    fi
done
for repository in "${tsml_eval_dir}" "${aeon_dir}"; do
    if [[ ! -d "${repository}/.git" ]]; then
        echo "ERROR: Git checkout is missing: ${repository}" >&2
        exit 1
    fi
done
for problem in "${problems[@]}"; do
    for split in TRAIN TEST; do
        source_file="${data_root}/${problem}/${problem}_${split}.ts"
        if [[ ! -s "${source_file}" && ! -s "${source_file}.ts" ]]; then
            echo "ERROR: missing ${problem} ${split} data." >&2
            exit 1
        fi
    done
done

mkdir -p "${transform_root}" "${output_root}" "${numba_cache_dir}"

tsml_eval_commit=$(git -C "${tsml_eval_dir}" rev-parse HEAD)
aeon_commit=$(git -C "${aeon_dir}" rev-parse HEAD)

PYTHONNOUSERSITE=1 PYTHONPATH="${aeon_dir}:${tsml_eval_dir}" \
"${python_path}" - <<'PY'
import aeon
import aeon_neuro
import tsml_eval
from tsml_eval._wip.eeg_cote.generate_archive_transform import _make_transformer

print("aeon:     ", aeon.__file__)
print("aeon-neuro:", aeon_neuro.__file__)
print("tsml-eval:", tsml_eval.__file__)
print("transform:", type(_make_transformer("GEAR-Auto", 64, 0)).__name__)
PY

run_id=$(date +%Y%m%d%H%M%S)
submission_dir="${transform_root}/batch-submissions/${run_id}"
command_file="${submission_dir}/generatedCommandList-${run_id}.txt"
submission_file="${submission_dir}/generatedSubmissionFile-${run_id}.sub"
mkdir -p "${submission_dir}"
: > "${command_file}"

command_count=0
for problem in "${problems[@]}"; do
    destination="${transform_root}/${transform_name}/${problem}"
    if [[ -s "${destination}/${problem}_TRAIN.ts" \
          && -s "${destination}/${problem}_TEST.ts" \
          && -s "${destination}/transform_summary.json" ]]; then
        echo "Skipping complete transform: ${transform_name}/${problem}"
        continue
    fi

    log_dir="${output_root}/${problem}"
    mkdir -p "${log_dir}"
    log_file="${log_dir}/output-${run_id}.txt"
    command=(
        "${python_path}" -u "${worker}"
        --input-root "${data_root}"
        --output-root "${transform_root}"
        --problem "${problem}"
        --transform "${transform_name}"
        --random-state 0
    )
    printf -v command_line '%q ' "${command[@]}"
    printf '%s> %q 2>&1\n' "${command_line}" "${log_file}" >> "${command_file}"
    command_count=$((command_count + 1))
done

if ((command_count == 0)); then
    echo "Both missing GEAR-Auto transforms already exist; no job submitted."
    exit 0
fi

# One independent process per transform. There are currently only two cells.
cpu_count=${command_count}

cat > "${submission_file}" <<EOF
#!/bin/bash
#SBATCH --mail-type=${mail}
#SBATCH --mail-user=${mailto}
#SBATCH --job-name=eeg-gear-transform-missing
#SBATCH --partition=${queue}
#SBATCH --time=${max_time}
#SBATCH --output=${submission_dir}/%A-${run_id}.out
#SBATCH --error=${submission_dir}/%A-${run_id}.err
#SBATCH --nodes=1
#SBATCH --ntasks=${cpu_count}
#SBATCH --mem-per-cpu=${memory_per_cpu}

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

echo "Host:             \$(hostname)"
echo "Slurm job ID:     \${SLURM_JOB_ID}"
echo "Allocated tasks:  \${SLURM_NTASKS}"
echo "Transform cells:  ${command_count}"
echo "Memory per task:  ${memory_per_cpu}"
echo "Transform root:   ${transform_root}"
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
echo "Submitted ${command_count} transform cells using ${cpu_count} CPUs."
echo "Transformed data will be written below ${transform_root}/${transform_name}."
rm -f "${submission_file}"
