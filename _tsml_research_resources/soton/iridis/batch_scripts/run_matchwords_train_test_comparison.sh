#!/bin/bash

set -euo pipefail

# Run a selected stage of the end-to-end participant-disjoint MatchWords
# TRAIN/TEST case-study comparison. Each estimator receives its own Slurm job.

# ==============================================================================
# Configuration
# ==============================================================================

queue="batch"
max_num_submitted=200

username="ajb2u23"
mail="NONE"
mailto="${username}@soton.ac.uk"
local_path="/iridisfs/home/${username}"

problem="MatchWords"
resample=0

# Change this hard-coded value to select the submission stage:
#   "mrhydra"   : submit only MrHydra
#   "reductions": submit TSelect-HC2 and GMARv3-HC2
#   "all"       : submit HC2 and all three comparisons
run_set="mrhydra"

# classifier|memory|wall-time
case "${run_set}" in
    mrhydra)
        experiments=(
            "MRHydra|30G|12:00:00"
        )
        ;;
    reductions)
        experiments=(
            "TSelect-HC2|200G|60:00:00"
            "GMARv3-HC2|200G|60:00:00"
        )
        ;;
    all)
        experiments=(
            "HC2|300G|60:00:00"
            "MRHydra|30G|12:00:00"
            "TSelect-HC2|200G|60:00:00"
            "GMARv3-HC2|200G|60:00:00"
        )
        ;;
    *)
        echo "ERROR: unknown run_set: ${run_set}"
        echo "Use mrhydra, reductions, or all."
        exit 1
        ;;
esac

# ==============================================================================
# Repository, environment, data, and output locations
# ==============================================================================

tsml_eval_dir="${local_path}/Code/tsml-eval"
aeon_dir="${local_path}/Code/aeon"
experiment_script="${tsml_eval_dir}/tsml_eval/experiments/classification_experiments.py"

env_name="tsml-eval"
python_path="/home/${username}/.conda/envs/${env_name}/bin/python"

source_data_root="${local_path}/Data/EEG"
results_root="${local_path}/Results/MatchWordsCaseStudy/train-test"
output_root="${results_root}/output"
numba_cache_dir="${local_path}/Code/.cache/${env_name}"

# ==============================================================================
# Validation and standard filename aliases
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
for required_command in sbatch squeue; do
    if ! command -v "${required_command}" > /dev/null 2>&1; then
        echo "ERROR: required cluster command is unavailable: ${required_command}"
        exit 1
    fi
done

mkdir -p "${results_root}" "${output_root}" "${numba_cache_dir}"

source_problem="${source_data_root}/${problem}"
standard_train="${source_problem}/${problem}_TRAIN.ts"
standard_test="${source_problem}/${problem}_TEST.ts"
duplicated_train="${standard_train}.ts"
duplicated_test="${standard_test}.ts"
data_root="${source_data_root}"

if [[ ! -s "${standard_train}" || ! -s "${standard_test}" ]]; then
    if [[ ! -s "${duplicated_train}" || ! -s "${duplicated_test}" ]]; then
        echo "ERROR: MatchWords TRAIN/TEST files are missing."
        exit 1
    fi

    data_root="${results_root}/input-links"
    link_problem="${data_root}/${problem}"
    mkdir -p "${link_problem}"
    ln -sfn "${duplicated_train}" "${link_problem}/${problem}_TRAIN.ts"
    ln -sfn "${duplicated_test}" "${link_problem}/${problem}_TEST.ts"
fi

tsml_eval_commit=$(git -C "${tsml_eval_dir}" rev-parse HEAD)
aeon_commit=$(git -C "${aeon_dir}" rev-parse HEAD)
tsml_eval_branch=$(git -C "${tsml_eval_dir}" branch --show-current)
aeon_branch=$(git -C "${aeon_dir}" branch --show-current)

classifiers=()
for specification in "${experiments[@]}"; do
    IFS="|" read -r classifier memory walltime <<< "${specification}"
    if [[ -z "${classifier}" || -z "${memory}" || -z "${walltime}" ]]; then
        echo "ERROR: malformed experiment specification: ${specification}"
        exit 1
    fi
    classifiers+=("${classifier}")
done

echo "Problem:           ${problem}"
echo "Run set:           ${run_set}"
echo "Cases:             360 TRAIN / 360 TEST"
echo "Participant split: 9 TRAIN / 9 TEST"
echo "Classifiers:       ${classifiers[*]}"
echo "Data root:         ${data_root}"
echo "Results:           ${results_root}"
echo "tsml-eval branch:  ${tsml_eval_branch}"
echo "tsml-eval commit:  ${tsml_eval_commit}"
echo "aeon branch:       ${aeon_branch}"
echo "aeon commit:       ${aeon_commit}"
echo

PYTHONNOUSERSITE=1 \
PYTHONPATH="${aeon_dir}:${tsml_eval_dir}" \
"${python_path}" - "${classifiers[@]}" <<'PY'
import sys

import aeon
import tsml_eval
from tsml_eval.experiments import get_classifier_by_name

print("Python:   ", sys.executable)
print("aeon:     ", aeon.__file__)
print("tsml-eval:", tsml_eval.__file__)

for name in sys.argv[1:]:
    classifier = get_classifier_by_name(name, random_state=0, n_jobs=1)
    print(f"{name}: {type(classifier).__name__}")

print("Factory check succeeded")
PY

# ==============================================================================
# Submit one independent job per incomplete estimator
# ==============================================================================

run_id=$(date +%Y%m%d%H%M%S)
submission_dir="${results_root}/batch-submissions/${run_id}"
mkdir -p "${submission_dir}"

submitted=0
skipped=0

wait_for_queue_slot() {
    local num_jobs

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
            return
        fi
        echo "Waiting 60 seconds: ${num_jobs} jobs are running or pending."
        sleep 60
    done
}

for specification in "${experiments[@]}"; do
    IFS="|" read -r classifier memory walltime <<< "${specification}"

    result_file="${results_root}/${classifier}/Predictions/${problem}/testResample${resample}.csv"
    if [[ -s "${result_file}" ]]; then
        echo "${classifier}: complete result exists; skipping."
        skipped=$((skipped + 1))
        continue
    fi

    safe_classifier="${classifier//[^[:alnum:]]/-}"
    job_name="meg-matchwords-${safe_classifier}"
    classifier_output="${output_root}/${classifier}"
    experiment_output="${classifier_output}/output-${run_id}.txt"
    submission_file="${submission_dir}/generatedSubmissionFile-${safe_classifier}-${run_id}.sub"

    mkdir -p "${classifier_output}"

    cat > "${submission_file}" <<EOF
#!/bin/bash
#SBATCH --mail-type=${mail}
#SBATCH --mail-user=${mailto}
#SBATCH --job-name=${job_name}
#SBATCH --partition=${queue}
#SBATCH --time=${walltime}
#SBATCH --output=${submission_dir}/%A-${safe_classifier}-${run_id}.out
#SBATCH --error=${submission_dir}/%A-${safe_classifier}-${run_id}.err
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
    exit 1
fi
if [[ "\${current_aeon_commit}" != "${aeon_commit}" ]]; then
    echo "ERROR: aeon changed after submission."
    exit 1
fi

echo "Classifier:       ${classifier}"
echo "Host:             \$(hostname)"
echo "Slurm job ID:     \${SLURM_JOB_ID}"
echo "Memory request:   ${memory}"
echo "Wall time:        ${walltime}"
echo "tsml-eval commit: \${current_tsml_eval_commit}"
echo "aeon commit:      \${current_aeon_commit}"
echo

# Resample zero preserves the supplied participant-disjoint TRAIN/TEST split.
# Do not add -pr: that option expects MatchWords0_TRAIN.ts and _TEST.ts.
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

    wait_for_queue_slot
    if ! sbatch_output=$(sbatch "${submission_file}"); then
        echo "ERROR: failed to submit ${classifier}." >&2
        exit 1
    fi

    submitted=$((submitted + 1))
    echo "${sbatch_output}"
    echo "Submitted ${classifier}: memory=${memory}, wall-time=${walltime}"

    # Slurm has copied the submission program; no command-list file is needed.
    rm -f "${submission_file}"
done

echo
echo "Finished MatchWords train/test submissions."
echo "Submitted:          ${submitted}"
echo "Already complete:   ${skipped}"
echo "Results:            ${results_root}"
echo "Submission records: ${submission_dir}"
