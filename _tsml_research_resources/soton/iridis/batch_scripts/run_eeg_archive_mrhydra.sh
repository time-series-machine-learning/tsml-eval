#!/bin/bash

set -euo pipefail

# Run MrHydra once on each of the 25 EEG archive problems used in the paper.
# Results are written beside the channel-selection pipelines so the same
# evaluation scripts can use MrHydra as an external context classifier.

# ==============================================================================
# Experiment and Slurm configuration
# ==============================================================================

username="ajb2u23"
mail="NONE"
mailto="${username}@soton.ac.uk"

queue="batch"
max_time="60:00:00"
max_num_submitted=200

# Match the successful OpenCloseFist LOSO MrHydra configuration.
max_cpus_to_use=20
memory_per_cpu_gib=30
memory_per_cpu="${memory_per_cpu_gib}G"

local_path="/iridisfs/home/${username}"
job_name="eeg-mrhydra-context"

classifier="MrHydra"
resample=0

datasets=(
    "Alzheimers"
    "ButtonPress"
    "EyesOpenShut"
    "FaceDetection"
    "FeedbackButton"
    "FeetHands"
    "FibroLiverpool"
    "FibroUEA"
    "FingerMovements"
    "HandMovementDirection"
    "ImaginedFeetHands"
    "ImaginedOpenCloseFist"
    "InnerSpeech"
    "LongIntervalTask"
    "LowCost"
    "MatchingPennies"
    "MindReading"
    "MotorImagery"
    "OpenCloseFist"
    "PhotoStimulation"
    "PronouncedSpeech"
    "ShortIntervalTask"
    "SitStand"
    "SongFamiliarity"
    "VisualSpeech"
)

# ==============================================================================
# Repository, environment, data, and result locations
# ==============================================================================

tsml_eval_dir="${local_path}/Code/tsml-eval"
aeon_dir="${local_path}/Code/aeon"

script_file_path="${tsml_eval_dir}/tsml_eval/experiments/classification_experiments.py"

env_name="tsml-eval"
python_path="/home/${username}/.conda/envs/${env_name}/bin/python"

data_dir="${local_path}/Data/EEG"
results_dir="${local_path}/Results/ChannelSelectionPipeline"
out_dir="${results_dir}/output/${classifier}"
numba_cache_dir="${local_path}/Code/.cache/${env_name}"

# ==============================================================================
# Validate configuration and exact source state
# ==============================================================================

if [[ ! -x "${python_path}" ]]; then
    echo "ERROR: Python executable not found or not executable:"
    echo "  ${python_path}"
    exit 1
fi
if [[ ! -f "${script_file_path}" ]]; then
    echo "ERROR: classification experiment script not found:"
    echo "  ${script_file_path}"
    exit 1
fi
for repository in "${tsml_eval_dir}" "${aeon_dir}"; do
    if [[ ! -d "${repository}/.git" ]]; then
        echo "ERROR: Git checkout not found:"
        echo "  ${repository}"
        exit 1
    fi
done
if ((max_cpus_to_use < 1 || max_cpus_to_use > 192)); then
    echo "ERROR: max_cpus_to_use must be between 1 and 192."
    exit 1
fi
if ((max_cpus_to_use * memory_per_cpu_gib > 620)); then
    echo "ERROR: requested task memory exceeds the 620 GiB safety ceiling."
    exit 1
fi

for dataset in "${datasets[@]}"; do
    train_file="${data_dir}/${dataset}/${dataset}_TRAIN.ts"
    test_file="${data_dir}/${dataset}/${dataset}_TEST.ts"
    if [[ ! -s "${train_file}" || ! -s "${test_file}" ]]; then
        echo "ERROR: missing or empty data for ${dataset}."
        echo "  ${train_file}"
        echo "  ${test_file}"
        exit 1
    fi
done

mkdir -p "${results_dir}" "${out_dir}" "${numba_cache_dir}"

tsml_eval_commit=$(git -C "${tsml_eval_dir}" rev-parse HEAD)
aeon_commit=$(git -C "${aeon_dir}" rev-parse HEAD)
tsml_eval_branch=$(git -C "${tsml_eval_dir}" branch --show-current)
aeon_branch=$(git -C "${aeon_dir}" branch --show-current)

unset PYTHONHOME
export PYTHONNOUSERSITE=1
export PYTHONPATH="${aeon_dir}:${tsml_eval_dir}"

"${python_path}" - <<'PY'
import sys

import aeon
import tsml_eval
from tsml_eval.experiments import get_classifier_by_name

classifier = get_classifier_by_name("MrHydra", random_state=0, n_jobs=1)
print("Python:   ", sys.executable)
print("aeon:     ", aeon.__file__)
print("tsml-eval:", tsml_eval.__file__)
print("MrHydra:  ", classifier.__class__)
PY

echo "Classifier:        ${classifier}"
echo "Datasets:          ${#datasets[@]}"
echo "Maximum CPUs:      ${max_cpus_to_use}"
echo "Memory per CPU:    ${memory_per_cpu}"
echo "Maximum task RAM:  $((max_cpus_to_use * memory_per_cpu_gib)) GiB"
echo "Results:           ${results_dir}"
echo "tsml-eval branch:  ${tsml_eval_branch}"
echo "tsml-eval commit:  ${tsml_eval_commit}"
echo "aeon branch:       ${aeon_branch}"
echo "aeon commit:       ${aeon_commit}"
echo

# ==============================================================================
# Generate commands for results not already present
# ==============================================================================

run_id=$(date +%Y%m%d%H%M%S)
submission_dir="${results_dir}/batch-submissions/${run_id}"
command_file="${submission_dir}/generatedCommandList-${run_id}-mrhydra.txt"
submission_file="${submission_dir}/generatedSubmissionFile-${run_id}-mrhydra.sub"

mkdir -p "${submission_dir}"
: > "${command_file}"

command_count=0
for dataset in "${datasets[@]}"; do
    result_file="${results_dir}/${classifier}/Predictions/${dataset}/testResample${resample}.csv"
    if [[ -s "${result_file}" ]]; then
        echo "Skipping complete: ${classifier}/${dataset}/resample${resample}"
        continue
    fi

    experiment_output="${out_dir}/output-${dataset}-${resample}-${run_id}.txt"
    command=(
        "${python_path}"
        -u
        "${script_file_path}"
        "${data_dir}"
        "${results_dir}"
        "${classifier}"
        "${dataset}"
        "${resample}"
    )

    printf -v command_line '%q ' "${command[@]}"
    printf '%s> %q 2>&1\n' \
        "${command_line}" \
        "${experiment_output}" \
        >> "${command_file}"
    command_count=$((command_count + 1))
done

if ((command_count == 0)); then
    echo "All 25 MrHydra archive results already exist; nothing submitted."
    exit 0
fi

cpu_count=${command_count}
if ((cpu_count > max_cpus_to_use)); then
    cpu_count=${max_cpus_to_use}
fi
total_memory_gib=$((cpu_count * memory_per_cpu_gib))

# ==============================================================================
# Build and submit one task-farm job
# ==============================================================================

cat > "${submission_file}" <<EOF
#!/bin/bash
#SBATCH --mail-type=${mail}
#SBATCH --mail-user=${mailto}
#SBATCH --job-name=${job_name}
#SBATCH --partition=${queue}
#SBATCH --time=${max_time}
#SBATCH --output=${submission_dir}/%A-${run_id}-mrhydra.out
#SBATCH --error=${submission_dir}/%A-${run_id}-mrhydra.err
#SBATCH --nodes=1
#SBATCH --ntasks=${cpu_count}
#SBATCH --mem-per-cpu=${memory_per_cpu}

. /etc/profile
set -e

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
export MEMRECORD_INTERVAL=30
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

echo "Classifier:       ${classifier}"
echo "Host:             \$(hostname)"
echo "Slurm job ID:     \${SLURM_JOB_ID}"
echo "Allocated tasks:  \${SLURM_NTASKS}"
echo "Command count:    ${command_count}"
echo "Memory per CPU:   ${memory_per_cpu}"
echo "Command file:     ${command_file}"
echo "tsml-eval commit: \${current_tsml_eval_commit}"
echo "aeon commit:      \${current_aeon_commit}"
echo

"${python_path}" - <<'PY'
import sys

import aeon
import tsml_eval
from tsml_eval.experiments import get_classifier_by_name

classifier = get_classifier_by_name("MrHydra", random_state=0, n_jobs=1)
print("Python:   ", sys.executable)
print("aeon:     ", aeon.__file__)
print("tsml-eval:", tsml_eval.__file__)
print("MrHydra:  ", classifier.__class__)
print("Import and classifier-factory checks succeeded")
PY

staskfarm "${command_file}"
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
echo "Submitted MrHydra EEG archive task farm:"
echo "  Outstanding commands: ${command_count}"
echo "  Requested CPUs:       ${cpu_count}"
echo "  Memory per CPU:       ${memory_per_cpu}"
echo "  Maximum task RAM:     ${total_memory_gib} GiB"
echo "  Wall time:            ${max_time}"
echo "  Results:              ${results_dir}"
echo "  Submission directory: ${submission_dir}"

# Slurm has copied the submission script. Keep the command list because
# staskfarm reads it when the allocation starts.
rm -f "${submission_file}"
