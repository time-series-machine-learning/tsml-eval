#!/bin/bash

set -euo pipefail

# Run HC2 on the saved GEAR-Auto representations created by
# run_missing_gear_transforms.sh. This stage never fits GEAR again. Its result
# files use the original GEAR-Auto-HC2 name, while transform timing and memory
# remain available separately in Results/Transforms.

username="ajb2u23"
local_path="/iridisfs/home/${username}"
queue="batch"
max_num_submitted=200
max_time="60:00:00"
memory_per_cpu="30G"
mail="NONE"
mailto="${username}@soton.ac.uk"

transform_name="GEAR-Auto"
classifier_name="HC2"
problem_to_run="${PROBLEM:-all}"
case "${problem_to_run}" in
    all)
        problems=("FaceDetection" "LongIntervalTask")
        ;;
    FaceDetection|LongIntervalTask)
        problems=("${problem_to_run}")
        ;;
    *)
        echo "ERROR: PROBLEM must be all, FaceDetection, or LongIntervalTask." >&2
        exit 2
        ;;
esac

tsml_eval_dir="${local_path}/Code/tsml-eval"
aeon_dir="${local_path}/Code/aeon"
python_path="/home/${username}/.conda/envs/tsml-eval/bin/python"
worker="${tsml_eval_dir}/tsml_eval/_wip/eeg_cote/run_transformed_archive_classifier.py"
required_aeon_hc2_commit="ed21ac50acc9c80c5ff2827a374a81a0d69debbc"

transform_root="${local_path}/Results/Transforms"
results_root="${local_path}/Results/ChannelSelectionPipeline"
output_root="${results_root}/output/${transform_name}-${classifier_name}"
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
if ! git -C "${aeon_dir}" merge-base --is-ancestor \
    "${required_aeon_hc2_commit}" HEAD; then
    echo "ERROR: aeon lacks the accelerated HC2 Arsenal SVD fallback." >&2
    echo "Required ancestor: ${required_aeon_hc2_commit}" >&2
    echo "Current commit:    $(git -C "${aeon_dir}" rev-parse HEAD)" >&2
    exit 1
fi

mkdir -p "${results_root}" "${output_root}" "${numba_cache_dir}"
tsml_eval_commit=$(git -C "${tsml_eval_dir}" rev-parse HEAD)
aeon_commit=$(git -C "${aeon_dir}" rev-parse HEAD)

PYTHONNOUSERSITE=1 PYTHONPATH="${aeon_dir}:${tsml_eval_dir}" \
"${python_path}" - <<'PY'
import aeon
import tsml_eval
from tsml_eval.experiments._get_classifier import _make_hc2_or_component

classifier = _make_hc2_or_component(
    component="hc2",
    random_state=0,
    n_jobs=1,
    fit_contract=0,
    kwargs={},
)
print("aeon:     ", aeon.__file__)
print("tsml-eval:", tsml_eval.__file__)
print("classifier:", type(classifier).__name__)
PY

run_id=$(date +%Y%m%d%H%M%S)
submission_dir="${results_root}/batch-submissions/${run_id}"
command_file="${submission_dir}/generatedCommandList-${run_id}.txt"
submission_file="${submission_dir}/generatedSubmissionFile-${run_id}.sub"
mkdir -p "${submission_dir}"
: > "${command_file}"

command_count=0
for problem in "${problems[@]}"; do
    transformed_dir="${transform_root}/${transform_name}/${problem}"
    if [[ ! -s "${transformed_dir}/${problem}_TRAIN.ts" \
          || ! -s "${transformed_dir}/${problem}_TEST.ts" \
          || ! -s "${transformed_dir}/transform_summary.json" ]]; then
        echo "Transform not yet complete; skipping classifier: ${problem}"
        continue
    fi

    result_file="${results_root}/${transform_name}-${classifier_name}/Predictions/${problem}/testResample0.csv"
    if [[ -s "${result_file}" ]]; then
        echo "Skipping complete result: ${transform_name}-${classifier_name}/${problem}"
        continue
    fi

    log_dir="${output_root}/${problem}"
    mkdir -p "${log_dir}"
    log_file="${log_dir}/output-transform-first-${run_id}.txt"
    command=(
        "${python_path}" -u "${worker}"
        --transform-root "${transform_root}"
        --results-root "${results_root}"
        --transform "${transform_name}"
        --problem "${problem}"
        --classifier "${classifier_name}"
        --random-state 0
    )
    printf -v command_line '%q ' "${command[@]}"
    printf '%s> %q 2>&1\n' "${command_line}" "${log_file}" >> "${command_file}"
    command_count=$((command_count + 1))
done

if ((command_count == 0)); then
    echo "No classifier cells are ready and incomplete; no job submitted."
    exit 0
fi

cpu_count=${command_count}

cat > "${submission_file}" <<EOF
#!/bin/bash
#SBATCH --mail-type=${mail}
#SBATCH --mail-user=${mailto}
#SBATCH --job-name=eeg-gear-classifier-missing
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
echo "Classifier cells: ${command_count}"
echo "Problem selector:  ${problem_to_run}"
echo "Memory per task:  ${memory_per_cpu}"
echo "Transform root:   ${transform_root}"
echo "Results root:     ${results_root}"
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
echo "Submitted ${command_count} transformed HC2 cells using ${cpu_count} CPUs."
echo "HC2 timing excludes GEAR; use each transform_summary.json for transform cost."
rm -f "${submission_file}"
