#!/bin/bash

set -euo pipefail

# Rerun the full-channel DrCIF component at the 500 trees HC2 uses internally.
#
# Why this exists. HC2 forces n_estimators=500 for its internal DrCIF, but
# DrCIFClassifier defaults to 200 when built standalone. The full-channel
# component runs under Results/ChannelSelection/Full/DrCIF were produced before
# tsml-eval commit c13d028 (2026-07-25) added _make_hc2_or_component, so they
# used 200. Every channel-selection pipeline result in ChannelSelectionPipeline,
# and every leave-one-subject-out result, already uses 500. Full is therefore
# the only mismatched cell, and it is the baseline every selector is compared
# against.
#
# Only DrCIF is affected. Arsenal, STC and TDE were checked against HC2's
# internal settings and match exactly, so they are not rerun.
#
# The drcif-500 factory option builds DrCIFClassifier(n_estimators=500), the
# same estimator the pipeline builds for the selectors.
#
# Train files are generated because the assembled HC2 takes its CAWPE weights
# from the component training estimates.

# ==============================================================================
# Experiment configuration
# ==============================================================================

# Resamples are zero-indexed internally:
# start_fold=1 and max_folds=1 runs resample 0.
max_folds=1
start_fold=1

max_num_submitted=200
queue="batch"

# 25 processes at 20 GiB each request 500 GiB of a standard red node. DrCIF at
# 500 trees needs roughly 2.5 times the runtime of the 200-tree results.
max_cpus_to_use="${max_cpus_to_use:-25}"
memory_per_cpu_gib="${memory_per_cpu_gib:-20}"
memory_per_cpu="${memory_per_cpu_gib}G"

username="ajb2u23"
mail="NONE"
mailto="${username}@soton.ac.uk"

local_path="/iridisfs/home/${username}"

classifier="drcif-500"
job_name_prefix="eeg-full-drcif500"
submission_label="FullDrCIF500"

generate_train_files="true"
predefined_folds="false"
normalise_data="false"

# ==============================================================================
# Datasets
# ==============================================================================

# The 25 archive problems with at least ten channels. The slow group is
# submitted first so the long jobs start earliest.
slow_datasets=(
    "LongIntervalTask"
    "MatchingPennies"
    "ShortIntervalTask"
    "SitStand"
)

fast_datasets=(
    "SongFamiliarity"
    "FeetHands"
    "ImaginedOpenCloseFist"
    "ImaginedFeetHands"
    "OpenCloseFist"
    "FaceDetection"
    "Alzheimers"
    "VisualSpeech"
    "InnerSpeech"
    "PronouncedSpeech"
    "MotorImagery"
    "PhotoStimulation"
    "FibroUEA"
    "FibroLiverpool"
    "MindReading"
    "FeedbackButton"
    "LowCost"
    "ButtonPress"
    "HandMovementDirection"
    "FingerMovements"
    "EyesOpenShut"
)

datasets=("${slow_datasets[@]}" "${fast_datasets[@]}")

# ==============================================================================
# Repository, data, and result locations
# ==============================================================================

tsml_eval_dir="${local_path}/Code/tsml-eval"
aeon_dir="${local_path}/Code/aeon"

script_file_path="${tsml_eval_dir}/tsml_eval/experiments/classification_experiments.py"

env_name="tsml-eval"
python_path="/home/${username}/.conda/envs/${env_name}/bin/python"

data_dir="${local_path}/Data/EEG"

# Results land beside the selector pipelines. The assembly step reads component
# directories by explicit path, so the folder need not follow the
# Selector-Component convention used by the pipeline runs.
results_dir="${local_path}/Results/ChannelSelectionPipeline"
out_dir="${results_dir}/output"
numba_cache_dir="${local_path}/Code/.cache/${env_name}"

# ==============================================================================
# Validate configuration
# ==============================================================================

if [[ ! -x "${python_path}" ]]; then
    echo "ERROR: Python executable not found or not executable:"
    echo "  ${python_path}"
    exit 1
fi

if [[ ! -f "${script_file_path}" ]]; then
    echo "ERROR: tsml-eval classification script not found:"
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

if ((start_fold < 1 || max_folds < start_fold)); then
    echo "ERROR: invalid fold range ${start_fold}..${max_folds}."
    exit 1
fi

if ((max_cpus_to_use < 1 || max_cpus_to_use > 192)); then
    echo "ERROR: max_cpus_to_use must be between 1 and 192."
    exit 1
fi

if ((${#datasets[@]} != 25)); then
    echo "ERROR: expected 25 EEG datasets, found ${#datasets[@]}."
    exit 1
fi

declare -A seen_datasets
for dataset in "${datasets[@]}"; do
    if [[ -n "${seen_datasets[${dataset}]+present}" ]]; then
        echo "ERROR: duplicate dataset: ${dataset}"
        exit 1
    fi
    seen_datasets["${dataset}"]=1

    train_data="${data_dir}/${dataset}/${dataset}_TRAIN.ts"
    test_data="${data_dir}/${dataset}/${dataset}_TEST.ts"
    if [[ ! -s "${train_data}" || ! -s "${test_data}" ]]; then
        echo "ERROR: missing or empty raw data for ${dataset}:"
        echo "  ${train_data}"
        echo "  ${test_data}"
        exit 1
    fi
done

mkdir -p "${results_dir}" "${out_dir}" "${numba_cache_dir}"

# Record exact source states at submission. Queued jobs refuse to run if either
# checkout moves before its allocation begins.
tsml_eval_commit=$(git -C "${tsml_eval_dir}" rev-parse HEAD)
aeon_commit=$(git -C "${aeon_dir}" rev-parse HEAD)

echo "Classifier:        ${classifier}"
echo "Datasets:          ${#datasets[@]}"
echo "Generate train:    ${generate_train_files}"
echo "Memory per CPU:    ${memory_per_cpu}"
echo "Maximum CPUs:      ${max_cpus_to_use}"
echo "tsml-eval commit:  ${tsml_eval_commit}"
echo "aeon commit:       ${aeon_commit}"
echo

# Confirm the factory name resolves and really gives 500 trees before
# submitting any job.
PYTHONNOUSERSITE=1 \
PYTHONPATH="${aeon_dir}:${tsml_eval_dir}" \
"${python_path}" - "${classifier}" <<'PY'
import sys

import aeon
import tsml_eval
from tsml_eval.experiments import get_classifier_by_name

print("Python:    ", sys.executable)
print("aeon:      ", aeon.__file__)
print("tsml-eval: ", tsml_eval.__file__)

estimator = get_classifier_by_name(sys.argv[1], random_state=0, n_jobs=1)
n_estimators = getattr(estimator, "n_estimators", None)
print(f"{sys.argv[1]}: {type(estimator).__name__} n_estimators={n_estimators}")
if n_estimators != 500:
    raise SystemExit(f"ERROR: expected 500 trees to match HC2, got {n_estimators}")
print("Factory check succeeded")
PY

# ==============================================================================
# Convert Boolean options into tsml-eval arguments
# ==============================================================================

generate_train_arg=""
predefined_folds_arg=""
normalise_data_arg=""

if [[ "${generate_train_files,,}" == "true" ]]; then
    generate_train_arg="-tr"
fi

if [[ "${predefined_folds,,}" == "true" ]]; then
    predefined_folds_arg="-pr"
fi

if [[ "${normalise_data,,}" == "true" ]]; then
    normalise_data_arg="-rn"
fi

# ==============================================================================
# Submission state and helpers
# ==============================================================================

run_id=$(date +%Y%m%d%H%M%S)
submission_dir="${results_dir}/batch-submissions/${run_id}"
mkdir -p "${submission_dir}"

total_commands=0
slurm_job_count=0

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
            break
        fi

        echo "Waiting 60 seconds: ${num_jobs} jobs are running or pending."
        sleep 60
    done
}

submit_batch() {
    local batch_label="$1"
    local walltime="$2"
    shift 2
    local -a batch_datasets=("$@")

    local batch_id="${run_id}-${submission_label}-${batch_label}"
    local command_file="${submission_dir}/generatedCommandList-${batch_id}.txt"
    local submission_file="${submission_dir}/generatedSubmissionFile-${batch_id}.sub"
    local cmd_count=0
    local cpu_count
    local dataset
    local resample
    local test_file
    local train_file
    local experiment_output
    local command_line
    local sbatch_output

    : > "${command_file}"
    mkdir -p "${out_dir}/${classifier}"

    for dataset in "${batch_datasets[@]}"; do
        for ((
            resample = start_fold - 1;
            resample < max_folds;
            resample++
        )); do
            test_file="${results_dir}/${classifier}/Predictions/${dataset}/testResample${resample}.csv"
            train_file="${results_dir}/${classifier}/Predictions/${dataset}/trainResample${resample}.csv"

            # Skip only non-empty completed results.
            if [[ -s "${test_file}" ]]; then
                if [[ -z "${generate_train_arg}" || -s "${train_file}" ]]; then
                    continue
                fi
            fi

            experiment_output="${out_dir}/${classifier}/output-${dataset}-${resample}-${batch_id}.txt"

            command_line="PYTHONNOUSERSITE=1"
            command_line+=" PYTHONPATH=${aeon_dir}:${tsml_eval_dir}"
            command_line+=" NUMBA_CACHE_DIR=${numba_cache_dir}"
            command_line+=" OMP_NUM_THREADS=1 MKL_NUM_THREADS=1"
            command_line+=" OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1"
            command_line+=" ${python_path} ${script_file_path}"
            command_line+=" ${data_dir} ${results_dir} ${classifier} ${dataset}"
            command_line+=" ${resample}"
            command_line+=" ${generate_train_arg} ${predefined_folds_arg}"
            command_line+=" ${normalise_data_arg}"
            command_line+=" > ${experiment_output} 2>&1"

            echo "${command_line}" >> "${command_file}"
            cmd_count=$((cmd_count + 1))
        done
    done

    if ((cmd_count == 0)); then
        echo "${batch_label}: nothing to run, all results already complete."
        rm -f "${command_file}"
        return
    fi

    cpu_count=$((cmd_count < max_cpus_to_use ? cmd_count : max_cpus_to_use))

    wait_for_queue_slot

    {
        echo "#!/bin/bash"
        echo "#SBATCH --job-name=${job_name_prefix}-${batch_label}"
        echo "#SBATCH --partition=${queue}"
        echo "#SBATCH --nodes=1"
        echo "#SBATCH --ntasks=${cpu_count}"
        echo "#SBATCH --cpus-per-task=1"
        echo "#SBATCH --mem-per-cpu=${memory_per_cpu}"
        echo "#SBATCH --time=${walltime}"
        echo "#SBATCH --output=${out_dir}/%x-%j.out"
        echo "#SBATCH --error=${out_dir}/%x-%j.err"
        echo "#SBATCH --mail-type=${mail}"
        echo "#SBATCH --mail-user=${mailto}"
        echo
        echo "set -euo pipefail"
        echo
        echo "# Refuse to run if either checkout moved between submission and"
        echo "# allocation."
        echo "actual=\$(git -C ${tsml_eval_dir} rev-parse HEAD)"
        echo "if [[ \"\${actual}\" != \"${tsml_eval_commit}\" ]]; then"
        echo "    echo \"ERROR: tsml-eval moved to \${actual}\""
        echo "    exit 1"
        echo "fi"
        echo "actual=\$(git -C ${aeon_dir} rev-parse HEAD)"
        echo "if [[ \"\${actual}\" != \"${aeon_commit}\" ]]; then"
        echo "    echo \"ERROR: aeon moved to \${actual}\""
        echo "    exit 1"
        echo "fi"
        echo
        echo "module load staskfarm"
        echo "staskfarm -v ${command_file}"
    } > "${submission_file}"

    sbatch_output=$(sbatch "${submission_file}")
    echo "${batch_label}: ${cmd_count} command(s) on ${cpu_count} CPU(s) -> ${sbatch_output}"

    total_commands=$((total_commands + cmd_count))
    slurm_job_count=$((slurm_job_count + 1))
}

submit_batch "slow" "60:00:00" "${slow_datasets[@]}"
submit_batch "fast" "60:00:00" "${fast_datasets[@]}"

echo
echo "Submitted ${slurm_job_count} SLURM job(s), ${total_commands} command(s)."
echo "Results:     ${results_dir}/${classifier}/Predictions"
echo "Submissions: ${submission_dir}"
echo
echo "When complete, rebuild full-channel HC2 with the corrected component:"
echo "  Arsenal, STC and TDE come from the existing Full runs,"
echo "  DrCIF comes from ${results_dir}/${classifier}."
