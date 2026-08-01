#!/bin/bash

set -euo pipefail

# Run one channel/time-reduction prefix with HC2 and all four HC2 components
# over the complete 25-problem EEG archive used in the paper.

# ==============================================================================
# Experiment configuration
# ==============================================================================

# Resamples are zero-indexed internally:
# start_fold=1 and max_folds=1 runs resample 0.
max_folds=1
start_fold=1

max_num_submitted=200
queue="batch"

# These defaults can be overridden by a small prefix-specific wrapper.
# 25 processes at 20 GiB each request 500 GiB of a standard red node.
max_cpus_to_use="${max_cpus_to_use:-25}"
memory_per_cpu_gib="${memory_per_cpu_gib:-20}"
memory_per_cpu="${memory_per_cpu_gib}G"

username="ajb2u23"
mail="NONE"
mailto="${username}@soton.ac.uk"

max_time="60:00:00"
local_path="/iridisfs/home/${username}"

classifier_prefix="${classifier_prefix:-GMARv3}"
job_name_prefix="${job_name_prefix:-eeg-gmarv3}"

generate_train_files="${generate_train_files:-false}"
predefined_folds="false"
normalise_data="false"

# Optional wrapper controls. Defaults preserve the original three-job archive
# run containing HC2 and all four components.
components_only="${components_only:-false}"
batch_mode="${batch_mode:-three_batches}"
component_to_run="${component_to_run:-}"
submission_label="${submission_label:-${classifier_prefix}}"

# ==============================================================================
# Runtime batches
# ==============================================================================

# Ordered using the established full-HC2 runtime estimates. The previously
# omitted ImaginedOpenCloseFist belongs in batch 2.
batch1_datasets=(
    "EyesOpenShut"
    "FingerMovements"
    "HandMovementDirection"
    "ButtonPress"
    "LowCost"
    "FeedbackButton"
    "MindReading"
    "FibroLiverpool"
    "FibroUEA"
    "PhotoStimulation"
)

batch2_datasets=(
    "MotorImagery"
    "PronouncedSpeech"
    "InnerSpeech"
    "VisualSpeech"
    "Alzheimers"
    "FaceDetection"
    "OpenCloseFist"
    "ImaginedFeetHands"
    "ImaginedOpenCloseFist"
    "FeetHands"
    "SongFamiliarity"
)

batch3_datasets=(
    "SitStand"
    "ShortIntervalTask"
    "MatchingPennies"
    "LongIntervalTask"
)

fast_datasets=(
    "${batch1_datasets[@]}"
    "${batch2_datasets[@]}"
)

slow_datasets=(
    "${batch3_datasets[@]}"
)

case "${batch_mode}" in
    three_batches)
        batch_labels=("batch1" "batch2" "batch3")
        batch_walltimes=("60:00:00" "60:00:00" "60:00:00")
        ;;
    fast_slow)
        batch_labels=("fast" "slow")
        batch_walltimes=("60:00:00" "60:00:00")
        ;;
    fast_only)
        batch_labels=("fast")
        batch_walltimes=("60:00:00")
        ;;
    slow_only)
        batch_labels=("slow")
        batch_walltimes=("60:00:00")
        ;;
    *)
        echo "ERROR: unknown batch_mode: ${batch_mode}"
        echo "Use three_batches, fast_slow, fast_only, or slow_only."
        exit 1
        ;;
esac

# ==============================================================================
# Repository, data, and result locations
# ==============================================================================

tsml_eval_dir="${local_path}/Code/tsml-eval"
aeon_dir="${local_path}/Code/aeon"

script_file_path="${tsml_eval_dir}/tsml_eval/experiments/classification_experiments.py"

env_name="tsml-eval"
python_path="/home/${username}/.conda/envs/${env_name}/bin/python"

data_dir="${local_path}/Data/EEG"
results_dir="${local_path}/Results/ChannelSelectionPipeline"
out_dir="${results_dir}/output"
numba_cache_dir="${local_path}/Code/.cache/${env_name}"

# ==============================================================================
# Pipeline variants
# ==============================================================================

if [[ -n "${component_to_run}" ]]; then
    case "${component_to_run}" in
        Arsenal|DrCIF|STC|HC2|TDE)
            components_to_run=("${component_to_run}")
            ;;
        *)
            echo "ERROR: unknown component_to_run: ${component_to_run}"
            echo "Use Arsenal, DrCIF, STC, HC2, or TDE."
            exit 1
            ;;
    esac
elif [[ "${components_only,,}" == "true" ]]; then
    components_to_run=(
        "Arsenal"
        "DrCIF"
        "STC"
        "TDE"
    )
else
    components_to_run=(
        "Arsenal"
        "DrCIF"
        "STC"
        "HC2"
        "TDE"
    )
fi

classifiers_to_run=()
for component in "${components_to_run[@]}"; do
    classifiers_to_run+=("${classifier_prefix}-${component}")
done

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

if ((memory_per_cpu_gib < 1)); then
    echo "ERROR: memory_per_cpu_gib must be positive."
    exit 1
fi

if (( ${#batch_labels[@]} != ${#batch_walltimes[@]} )); then
    echo "ERROR: batch_labels and batch_walltimes must have the same length."
    exit 1
fi

declare -A seen_datasets
dataset_count=0

for batch_label in "${batch_labels[@]}"; do
    case "${batch_label}" in
        batch1)
            datasets=("${batch1_datasets[@]}")
            ;;
        batch2)
            datasets=("${batch2_datasets[@]}")
            ;;
        batch3)
            datasets=("${batch3_datasets[@]}")
            ;;
        fast)
            datasets=("${fast_datasets[@]}")
            ;;
        slow)
            datasets=("${slow_datasets[@]}")
            ;;
        *)
            echo "ERROR: unknown batch label: ${batch_label}"
            exit 1
            ;;
    esac

    for dataset in "${datasets[@]}"; do
        if [[ -n "${seen_datasets[${dataset}]+present}" ]]; then
            echo "ERROR: dataset appears in more than one batch: ${dataset}"
            exit 1
        fi
        seen_datasets["${dataset}"]="${batch_label}"
        dataset_count=$((dataset_count + 1))

        train_data="${data_dir}/${dataset}/${dataset}_TRAIN.ts"
        test_data="${data_dir}/${dataset}/${dataset}_TEST.ts"
        if [[ ! -s "${train_data}" || ! -s "${test_data}" ]]; then
            echo "ERROR: missing or empty raw data for ${dataset}:"
            echo "  ${train_data}"
            echo "  ${test_data}"
            exit 1
        fi
    done
done

case "${batch_mode}" in
    three_batches|fast_slow)
        expected_dataset_count=25
        ;;
    fast_only)
        expected_dataset_count=21
        ;;
    slow_only)
        expected_dataset_count=4
        ;;
esac

if ((dataset_count != expected_dataset_count)); then
    echo "ERROR: expected ${expected_dataset_count} unique EEG datasets "
    echo "for ${batch_mode}, but found ${dataset_count}."
    exit 1
fi

mkdir -p "${results_dir}" "${out_dir}" "${numba_cache_dir}"

# Record exact source states at submission. Queued jobs refuse to run if either
# checkout moves before its allocation begins. aeon-neuro is installed in the
# conda environment and is deliberately not placed on PYTHONPATH here.
tsml_eval_commit=$(git -C "${tsml_eval_dir}" rev-parse HEAD)
aeon_commit=$(git -C "${aeon_dir}" rev-parse HEAD)

echo "Classifier prefix: ${classifier_prefix}"
echo "Pipeline variants: ${#classifiers_to_run[@]}"
echo "Datasets:          ${dataset_count}"
echo "Expected results:  $((${#classifiers_to_run[@]} * dataset_count))"
echo "Generate train:    ${generate_train_files}"
echo "Batch mode:        ${batch_mode}"
echo "Submission label:  ${submission_label}"
echo "Memory per CPU:    ${memory_per_cpu}"
echo "Maximum CPUs:      ${max_cpus_to_use}"
echo "tsml-eval commit:  ${tsml_eval_commit}"
echo "aeon commit:       ${aeon_commit}"
echo

# Confirm every configured factory name before submitting any job. aeon-neuro
# resolves from the active conda environment, while aeon and tsml-eval use
# these checkouts.
PYTHONNOUSERSITE=1 \
PYTHONPATH="${aeon_dir}:${tsml_eval_dir}" \
"${python_path}" - "${classifiers_to_run[@]}" <<'PY'
import sys

import aeon
import aeon_neuro
import tsml_eval
from tsml_eval.experiments import get_classifier_by_name

print("Python:     ", sys.executable)
print("aeon:       ", aeon.__file__)
print("aeon-neuro: ", aeon_neuro.__file__)
print("tsml-eval:  ", tsml_eval.__file__)

for classifier_name in sys.argv[1:]:
    estimator = get_classifier_by_name(
        classifier_name,
        random_state=0,
        n_jobs=1,
    )
    print(f"{classifier_name}: {type(estimator).__name__}")

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
    local -a datasets=("$@")

    local batch_id="${run_id}-${submission_label}-${batch_label}"
    local command_file="${submission_dir}/generatedCommandList-${batch_id}.txt"
    local submission_file="${submission_dir}/generatedSubmissionFile-${batch_id}.sub"
    local cmd_count=0
    local cpu_count
    local classifier
    local dataset
    local resample
    local test_file
    local train_file
    local experiment_output
    local command_line
    local sbatch_output
    local -a command

    : > "${command_file}"

    # Dataset is the outer loop, keeping the configured variants adjacent.
    for dataset in "${datasets[@]}"; do
        for classifier in "${classifiers_to_run[@]}"; do
            mkdir -p "${out_dir}/${classifier}"

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

                if [[ -n "${generate_train_arg}" ]]; then
                    command+=("${generate_train_arg}")
                fi
                if [[ -n "${predefined_folds_arg}" ]]; then
                    command+=("${predefined_folds_arg}")
                fi
                if [[ -n "${normalise_data_arg}" ]]; then
                    command+=("${normalise_data_arg}")
                fi

                printf -v command_line '%q ' "${command[@]}"
                printf '%s> %q 2>&1\n' \
                    "${command_line}" \
                    "${experiment_output}" \
                    >> "${command_file}"

                cmd_count=$((cmd_count + 1))
                total_commands=$((total_commands + 1))
            done
        done
    done

    if ((cmd_count == 0)); then
        echo "All ${classifier_prefix} results exist for ${batch_label}; no job submitted."
        return
    fi

    cpu_count=${cmd_count}
    if ((cpu_count > max_cpus_to_use)); then
        cpu_count=${max_cpus_to_use}
    fi

    cat > "${submission_file}" <<EOF
#!/bin/bash
#SBATCH --mail-type=${mail}
#SBATCH --mail-user=${mailto}
#SBATCH --job-name=${job_name_prefix}-${batch_label}
#SBATCH --partition=${queue}
#SBATCH --time=${walltime}
#SBATCH --output=${submission_dir}/%A-${batch_id}.out
#SBATCH --error=${submission_dir}/%A-${batch_id}.err
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

echo "Classifier prefix: ${classifier_prefix}"
echo "Batch:             ${batch_label}"
echo "Host:              \$(hostname)"
echo "Slurm job ID:      \${SLURM_JOB_ID}"
echo "Allocated tasks:   \${SLURM_NTASKS}"
echo "Command count:     ${cmd_count}"
echo "Memory per CPU:    ${memory_per_cpu}"
echo "tsml-eval commit:  \${current_tsml_eval_commit}"
echo "aeon commit:       \${current_aeon_commit}"
echo "Command file:      ${command_file}"
echo

"${python_path}" - <<'PY'
import sys

import aeon
import aeon_neuro
import tsml_eval

print("Python:     ", sys.executable)
print("aeon:       ", aeon.__file__)
print("aeon-neuro: ", aeon_neuro.__file__)
print("tsml-eval:  ", tsml_eval.__file__)
print("Import check succeeded")
PY

staskfarm "${command_file}"
EOF

    wait_for_queue_slot

    if ! sbatch_output=$(sbatch "${submission_file}"); then
        echo "ERROR: failed to submit ${submission_file}" >&2
        exit 1
    fi

    slurm_job_count=$((slurm_job_count + 1))

    echo "${sbatch_output}"
    echo "Submitted ${classifier_prefix} ${batch_label}:"
    echo "  Datasets:            ${#datasets[@]}"
    echo "  Outstanding commands: ${cmd_count}"
    echo "  Requested CPUs:       ${cpu_count}"
    echo "  Memory per CPU:       ${memory_per_cpu}"
    echo "  Maximum node memory:  $((cpu_count * memory_per_cpu_gib)) GiB"
    echo "  Wall time:            ${walltime}"
    echo

    # Slurm has copied the submission file. Retain the command list because
    # staskfarm reads it when the queued job begins.
    rm -f "${submission_file}"
}

# ==============================================================================
# Generate and submit one task farm per runtime batch
# ==============================================================================

for batch_index in "${!batch_labels[@]}"; do
    batch_label="${batch_labels[batch_index]}"
    walltime="${batch_walltimes[batch_index]}"

    case "${batch_label}" in
        batch1)
            datasets=("${batch1_datasets[@]}")
            ;;
        batch2)
            datasets=("${batch2_datasets[@]}")
            ;;
        batch3)
            datasets=("${batch3_datasets[@]}")
            ;;
        fast)
            datasets=("${fast_datasets[@]}")
            ;;
        slow)
            datasets=("${slow_datasets[@]}")
            ;;
    esac

    submit_batch "${batch_label}" "${walltime}" "${datasets[@]}"
done

echo "Finished submitting ${classifier_prefix}."
echo "Experiment commands submitted: ${total_commands}"
echo "Slurm jobs submitted:           ${slurm_job_count}"
echo "Submission directory:           ${submission_dir}"
