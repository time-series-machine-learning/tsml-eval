#!/bin/bash
# Submit recoverable missing Multiverse Core resample-0 results without overwriting.

set -euo pipefail

username=${USER:-ajb}
local_path="/gpfs/home/${username}"
repo_dir="${local_path}/Code/tsml-eval"
data_dir="${local_path}/Data/Multiverse"
dataset_file="${repo_dir}/_tsml_research_resources/dataset_lists/MultiverseCore.txt"
results_dir="${local_path}/Results/Multiverse/TestOnly/MultiverseCore"
output_dir="${results_dir}/output"
oom_file="${repo_dir}/_tsml_research_resources/oom_resample0.txt"

account="cmp"
partition="compute"
qos="uea-core-default"
max_active_tasks=200
standard_memory_mb=64000
oom_memory_mb=128000
time_limit="7-00:00:00"
env_name="tsml-eval"
module_name="python/anaconda/2024.10/3.12.7"
conda_sh="/gpfs/software/hali/python/anaconda/2024.10/etc/profile.d/conda.sh"
numba_cache_dir="${local_path}/Code/.cache/numba/${env_name}"

include_oom=false
if [[ "${1:-}" == "--include-oom" ]]; then
    include_oom=true
elif [[ $# -gt 0 ]]; then
    echo "Usage: bash $0 [--include-oom]" >&2
    exit 2
fi

classifiers=(
    "1NN-DTW"
    "Arsenal"
    "Catch22"
    "CIF"
    "DrCIF"
    "Dummy"
    "FreshPRINCE"
    "H-InceptionTime"
    "HC2"
    "LiteTIME"
    "MRHydra"
    "QUANT"
    "RDST"
    "RIST"
    "ROCKET"
    "STC"
    "TDE"
)

for required in "${repo_dir}" "${data_dir}" "${results_dir}"; do
    if [[ ! -d "${required}" ]]; then
        echo "ERROR: required directory not found: ${required}" >&2
        exit 1
    fi
done

for required in "${dataset_file}" "${oom_file}" "${conda_sh}"; do
    if [[ ! -f "${required}" ]]; then
        echo "ERROR: required file not found: ${required}" >&2
        exit 1
    fi
done

for command in git sbatch squeue; do
    if ! command -v "${command}" >/dev/null 2>&1; then
        echo "ERROR: ${command} was not found." >&2
        exit 1
    fi
done

repo_commit=$(git -C "${repo_dir}" rev-parse HEAD)
repo_branch=$(git -C "${repo_dir}" branch --show-current)
mkdir -p "${output_dir}" "${numba_cache_dir}"

declare -A oom_tasks=()
while IFS= read -r pair || [[ -n "${pair}" ]]; do
    pair=${pair//$'\r'/}
    [[ -z "${pair}" || "${pair}" == \#* ]] && continue
    oom_tasks["${pair}"]=1
done < "${oom_file}"

mapfile -t datasets < <(
    sed -e 's/\r$//' -e '/^[[:space:]]*$/d' -e '/^[[:space:]]*#/d' \
        "${dataset_file}"
)

if ((${#datasets[@]} == 0)); then
    echo "ERROR: no datasets found in ${dataset_file}" >&2
    exit 1
fi

queue_output=$(
    squeue --noheader --array --user="${username}" \
        --partition="${partition}" --states=RUNNING,PENDING \
        --format='%200j|%a|%T'
)

declare -A active_tasks=()
active_count=0
while IFS='|' read -r raw_name raw_index raw_state; do
    [[ -z "${raw_name}" ]] && continue
    job_name=${raw_name//[[:space:]]/}
    array_index=${raw_index//[[:space:]]/}
    active_tasks["${job_name}|${array_index}"]=1
    ((active_count += 1))
done <<< "${queue_output}"

capacity=$((max_active_tasks - active_count))
((capacity < 0)) && capacity=0

echo "Repository:       ${repo_dir}"
echo "Revision:         ${repo_branch:-DETACHED} ${repo_commit}"
echo "Results:          ${results_dir}"
echo "Existing tasks:   ${active_count}/${max_active_tasks}"
echo "Submission slots: ${capacity}"
echo "Normal memory:    ${standard_memory_mb} MB"
echo "OOM memory:       ${oom_memory_mb} MB"
echo "Include OOMs:     ${include_oom}"
echo

submitted=0
existing=0
active=0
deferred_oom=0
manual=0
errors=0
capacity_reached=false

for classifier in "${classifiers[@]}"; do
    for dataset in "${datasets[@]}"; do
        result_file="${results_dir}/${classifier}/Predictions/${dataset}/testResample0.csv"
        if [[ -e "${result_file}" ]]; then
            ((existing += 1))
            continue
        fi

        job_name="${classifier}_${dataset}"
        task_key="${job_name}|1"
        if [[ -n "${active_tasks[${task_key}]+present}" ]]; then
            echo "ACTIVE: ${classifier}/${dataset}; not duplicating"
            ((active += 1))
            continue
        fi

        if [[ "${classifier}/${dataset}" == "ROCKET/AustraliaRainfall_disc" ]]; then
            echo "MANUAL: ${classifier}/${dataset}; deterministic LAPACK overflow"
            ((manual += 1))
            continue
        fi

        if ((submitted >= capacity)); then
            capacity_reached=true
            break 2
        fi

        pair_key="${classifier}/${dataset}"
        if [[ -n "${oom_tasks[${pair_key}]+present}" &&
            "${include_oom}" == false ]]; then
            echo "DEFERRED OOM: ${classifier}/${dataset}"
            ((deferred_oom += 1))
            continue
        fi

        memory_mb=${standard_memory_mb}
        memory_label="standard retry"
        if [[ -n "${oom_tasks[${pair_key}]+present}" ]]; then
            memory_mb=${oom_memory_mb}
            memory_label="OOM retry"
        fi

        job_output_dir="${output_dir}/${classifier}/${dataset}"
        mkdir -p "${job_output_dir}"
        batch_file=$(mktemp "${output_dir}/resample0.XXXXXX.sub")

        cat > "${batch_file}" <<EOF
#!/bin/bash
#SBATCH --account=${account}
#SBATCH --partition=${partition}
#SBATCH --qos=${qos}
#SBATCH --time=${time_limit}
#SBATCH --job-name=${job_name}
#SBATCH --array=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=${memory_mb}M
#SBATCH --output=${job_output_dir}/%A-%a.out
#SBATCH --error=${job_output_dir}/%A-%a.err

set -eo pipefail
source /etc/profile
module purge
module load ${module_name}
source ${conda_sh}
conda activate ${env_name}

export NUMBA_CACHE_DIR="${numba_cache_dir}"
export CUDA_VISIBLE_DEVICES=""
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export MPI_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export LOKY_MAX_CPU_COUNT=1
export TF_NUM_INTEROP_THREADS=1
export TF_NUM_INTRAOP_THREADS=1
export PYTHONUNBUFFERED=1

if [[ -e "${result_file}" ]]; then
    echo "Result now exists; refusing to overwrite ${result_file}"
    exit 0
fi

cd "${repo_dir}"
actual_commit=\$(git rev-parse HEAD)
if [[ "\${actual_commit}" != "${repo_commit}" ]]; then
    echo "ERROR: repository changed after submission."
    echo "Expected: ${repo_commit}"
    echo "Actual:   \${actual_commit}"
    exit 1
fi

echo "Classifier:       ${classifier}"
echo "Dataset:          ${dataset}"
echo "Resample ID:      0"
echo "Requested memory: ${memory_mb} MB"
echo "CPU-only:         true"

python -u -m tsml_eval.experiments.classification_experiments \
    "${data_dir}" \
    "${results_dir}" \
    "${classifier}" \
    "${dataset}" \
    0
EOF

        if job_id=$(sbatch --parsable "${batch_file}"); then
            echo "SUBMITTED: ${classifier}/${dataset} as ${job_id} "\
                "(${memory_label}, ${memory_mb} MB)"
            ((submitted += 1))
            active_tasks["${task_key}"]=1
        else
            echo "ERROR: submission failed for ${classifier}/${dataset}" >&2
            ((errors += 1))
        fi
        rm -f -- "${batch_file}"
    done
done

echo
echo "Finished targeted resample-0 submission."
echo "Submitted:           ${submitted}"
echo "Existing files:      ${existing}"
echo "Already active:      ${active}"
echo "Deferred OOM:        ${deferred_oom}"
echo "Needs code change:   ${manual}"
echo "Submission failures: ${errors}"
if [[ "${capacity_reached}" == true ]]; then
    echo "Queue ceiling reached; run this script again after jobs finish."
fi
