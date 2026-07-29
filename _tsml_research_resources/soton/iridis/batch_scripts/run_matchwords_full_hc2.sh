#!/bin/bash

set -euo pipefail

# Run the unreduced HIVECOTEV2 baseline on the participant-disjoint MatchWords
# MEG split. This is deliberately separate from the transform task farm so its
# fit time and peak memory describe full HC2 on the raw data.

# ==============================================================================
# Experiment configuration
# ==============================================================================

queue="batch"
max_num_submitted=200
memory="300G"
max_time="60:00:00"

username="ajb2u23"
mail="NONE"
mailto="${username}@soton.ac.uk"
local_path="/iridisfs/home/${username}"
job_name="meg-matchwords-full-hc2"

problem="MatchWords"
classifier="HC2"
resample=0

# ==============================================================================
# Repository, environment, data, and result locations
# ==============================================================================

tsml_eval_dir="${local_path}/Code/tsml-eval"
aeon_dir="${local_path}/Code/aeon"
experiment_script="${tsml_eval_dir}/tsml_eval/experiments/classification_experiments.py"

env_name="tsml-eval"
python_path="/home/${username}/.conda/envs/${env_name}/bin/python"

source_data_root="${local_path}/Data/EEG"
results_root="${local_path}/Results/MatchWordsCaseStudy/full-hc2"
output_root="${results_root}/output"
numba_cache_dir="${local_path}/Code/.cache/${env_name}"

# ==============================================================================
# Validate and, if necessary, provide standard .ts filename aliases
# ==============================================================================

if [[ ! -x "${python_path}" ]]; then
    echo "ERROR: Python executable not found: ${python_path}"
    exit 1
fi
if [[ ! -f "${experiment_script}" ]]; then
    echo "ERROR: classification experiment program not found: ${experiment_script}"
    exit 1
fi
for repository in "${tsml_eval_dir}" "${aeon_dir}"; do
    if [[ ! -d "${repository}/.git" ]]; then
        echo "ERROR: Git checkout not found: ${repository}"
        exit 1
    fi
done

source_problem="${source_data_root}/${problem}"
standard_train="${source_problem}/${problem}_TRAIN.ts"
standard_test="${source_problem}/${problem}_TEST.ts"
duplicated_train="${standard_train}.ts"
duplicated_test="${standard_test}.ts"
data_root="${source_data_root}"

mkdir -p "${results_root}" "${output_root}" "${numba_cache_dir}"

# Some downloaded copies used .ts.ts. tsml-eval expects the standard names, so
# provide aliases under the result area without modifying or copying raw data.
if [[ ! -s "${standard_train}" || ! -s "${standard_test}" ]]; then
    if [[ ! -s "${duplicated_train}" || ! -s "${duplicated_test}" ]]; then
        echo "ERROR: MatchWords TRAIN/TEST files are missing."
        echo "Expected standard .ts files or the known .ts.ts variants."
        exit 1
    fi

    data_root="${results_root}/input-links"
    link_problem="${data_root}/${problem}"
    mkdir -p "${link_problem}"
    ln -sfn "${duplicated_train}" "${link_problem}/${problem}_TRAIN.ts"
    ln -sfn "${duplicated_test}" "${link_problem}/${problem}_TEST.ts"
fi

test_result="${results_root}/${classifier}/Predictions/${problem}/testResample${resample}.csv"
if [[ -s "${test_result}" ]]; then
    echo "Full HC2 result already exists; no job submitted:"
    echo "  ${test_result}"
    exit 0
fi

tsml_eval_commit=$(git -C "${tsml_eval_dir}" rev-parse HEAD)
aeon_commit=$(git -C "${aeon_dir}" rev-parse HEAD)
tsml_eval_branch=$(git -C "${tsml_eval_dir}" branch --show-current)
aeon_branch=$(git -C "${aeon_dir}" branch --show-current)

echo "Problem:           ${problem}"
echo "Classifier:        ${classifier}"
echo "Data root:         ${data_root}"
echo "Results:           ${results_root}"
echo "Memory:            ${memory}"
echo "Wall time:         ${max_time}"
echo "tsml-eval branch:  ${tsml_eval_branch}"
echo "tsml-eval commit:  ${tsml_eval_commit}"
echo "aeon branch:       ${aeon_branch}"
echo "aeon commit:       ${aeon_commit}"
echo

# Verify that the exact checkout used for submission can build full HC2.
PYTHONNOUSERSITE=1 \
PYTHONPATH="${aeon_dir}:${tsml_eval_dir}" \
"${python_path}" - <<'PY'
import aeon
import tsml_eval
from tsml_eval.experiments._get_classifier import get_classifier_by_name

classifier = get_classifier_by_name("HC2", random_state=0, n_jobs=1)
print("aeon:     ", aeon.__file__)
print("tsml-eval:", tsml_eval.__file__)
print("HC2:      ", type(classifier).__name__)
print("Factory check succeeded")
PY

# ==============================================================================
# Build and submit the single experiment
# ==============================================================================

run_id=$(date +%Y%m%d%H%M%S)
submission_dir="${results_root}/batch-submissions/${run_id}"
submission_file="${submission_dir}/generatedSubmissionFile-${run_id}.sub"
experiment_output="${output_root}/output-${problem}-${resample}-${run_id}.txt"

mkdir -p "${submission_dir}"

cat > "${submission_file}" <<EOF
#!/bin/bash
#SBATCH --mail-type=${mail}
#SBATCH --mail-user=${mailto}
#SBATCH --job-name=${job_name}
#SBATCH --partition=${queue}
#SBATCH --time=${max_time}
#SBATCH --output=${submission_dir}/%A-${run_id}.out
#SBATCH --error=${submission_dir}/%A-${run_id}.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=${memory}

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
    echo "ERROR: tsml-eval changed after submission."
    echo "Expected: ${tsml_eval_commit}"
    echo "Current:  \${current_tsml_eval_commit}"
    exit 1
fi
if [[ "\${current_aeon_commit}" != "${aeon_commit}" ]]; then
    echo "ERROR: aeon changed after submission."
    echo "Expected: ${aeon_commit}"
    echo "Current:  \${current_aeon_commit}"
    exit 1
fi

echo "Host:             \$(hostname)"
echo "Slurm job ID:     \${SLURM_JOB_ID}"
echo "Memory request:   ${memory}"
echo "tsml-eval commit: \${current_tsml_eval_commit}"
echo "aeon commit:      \${current_aeon_commit}"
echo

# resample 0 uses the supplied TRAIN/TEST split unchanged. Do not add -pr:
# that option asks for MatchWords0_TRAIN.ts/MatchWords0_TEST.ts instead.
"${python_path}" \
    -u \
    "${experiment_script}" \
    "${data_root}" \
    "${results_root}" \
    "${classifier}" \
    "${problem}" \
    "${resample}" \
    > "${experiment_output}" 2>&1
EOF

while true; do
    num_jobs=$(
        squeue \
            --noheader \
            --user="${username}" \
            --partition="${queue}" \
            --states=RUNNING,PENDING |
            wc -l
    )
    if ((num_jobs < max_num_submitted)); then
        break
    fi
    echo "Waiting 60 seconds: ${num_jobs} jobs are running or pending."
    sleep 60
done

if ! sbatch_output=$(sbatch "${submission_file}"); then
    echo "ERROR: failed to submit ${submission_file}" >&2
    exit 1
fi

echo "${sbatch_output}"
echo "Submitted raw MatchWords full-HC2 baseline:"
echo "  Memory:             ${memory}"
echo "  Wall time:          ${max_time}"
echo "  Experiment log:     ${experiment_output}"
echo "  Expected result:    ${test_result}"
echo "  Submission records: ${submission_dir}"

rm -f "${submission_file}"
